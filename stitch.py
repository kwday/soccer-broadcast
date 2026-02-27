"""
stitch.py — Batch panorama stitching.

Auto-detects timecode vs audio sync, loads per-game calibration (or runs
calibrate.py), then batch warps + blends all frames from two camera videos
into a single stitched panoramic video.

Usage:
    python stitch.py --left left.mp4 --right right.mp4 --output stitched.mp4
    python stitch.py --left left.mp4 --right right.mp4 --cal calibrations/2026-03-15_cal.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date

import cv2
import numpy as np

from calibrate import calibrate, extract_frame


def detect_timecode_offset(left_path: str, right_path: str) -> float | None:
    """
    Attempt to detect timecode sync between two video files using ffprobe.

    Returns the offset in seconds, or None if timecode is not available.
    """
    def get_timecode(path):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_entries", "format_tags=timecode:stream_tags=timecode",
                 path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)

            # Check format tags
            tc = data.get("format", {}).get("tags", {}).get("timecode")
            if tc:
                return tc

            # Check stream tags
            for stream in data.get("streams", []):
                tc = stream.get("tags", {}).get("timecode")
                if tc:
                    return tc

            return None
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            return None

    def timecode_to_seconds(tc_str: str, fps: float = 30.0) -> float:
        """Convert HH:MM:SS:FF or HH:MM:SS.mmm to seconds."""
        parts = tc_str.replace(";", ":").split(":")
        if len(parts) == 4:
            h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            return h * 3600 + m * 60 + s + f / fps
        elif len(parts) == 3:
            h, m = int(parts[0]), int(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
        return None

    tc_left = get_timecode(left_path)
    tc_right = get_timecode(right_path)

    if tc_left is None or tc_right is None:
        return None

    sec_left = timecode_to_seconds(tc_left)
    sec_right = timecode_to_seconds(tc_right)

    if sec_left is None or sec_right is None:
        return None

    return sec_right - sec_left


def get_video_info(path: str) -> dict:
    """Get video metadata using OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def load_calibration(cal_path: str) -> dict:
    """Load calibration data from JSON file."""
    with open(cal_path) as f:
        return json.load(f)


def stitch_frame(frame_left: np.ndarray, frame_right: np.ndarray,
                 H: np.ndarray, canvas_width: int, canvas_height: int,
                 offset_x: int, offset_y: int,
                 blend_x_start: int, blend_x_end: int) -> np.ndarray:
    """
    Stitch a single frame pair using the precomputed homography.

    Args:
        frame_left: Left camera frame (BGR).
        frame_right: Right camera frame (BGR).
        H: 3x3 homography matrix (maps right -> left space).
        canvas_width: Output canvas width.
        canvas_height: Output canvas height.
        offset_x: X translation to keep all pixels positive.
        offset_y: Y translation to keep all pixels positive.
        blend_x_start: Start of blend region in canvas coords.
        blend_x_end: End of blend region in canvas coords.

    Returns:
        Stitched panorama frame (BGR).
    """
    # Translation matrix to shift everything into positive coordinates
    T = np.array([[1, 0, -offset_x],
                  [0, 1, -offset_y],
                  [0, 0, 1]], dtype=np.float64)

    # Warp right image into canvas space
    H_adjusted = T @ H
    warped_right = cv2.warpPerspective(frame_right, H_adjusted,
                                       (canvas_width, canvas_height))

    # Place left image on canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    left_x = -offset_x
    left_y = -offset_y
    h_left, w_left = frame_left.shape[:2]

    # Compute valid placement region
    y_start = max(0, left_y)
    y_end = min(canvas_height, left_y + h_left)
    x_start = max(0, left_x)
    x_end = min(canvas_width, left_x + w_left)

    src_y_start = y_start - left_y
    src_y_end = y_end - left_y
    src_x_start = x_start - left_x
    src_x_end = x_end - left_x

    canvas[y_start:y_end, x_start:x_end] = \
        frame_left[src_y_start:src_y_end, src_x_start:src_x_end]

    # Linear blend in overlap region
    if blend_x_start < blend_x_end and blend_x_start >= 0:
        blend_w = blend_x_end - blend_x_start
        for i in range(blend_w):
            alpha = i / max(blend_w - 1, 1)
            x = blend_x_start + i
            if 0 <= x < canvas_width:
                left_col = canvas[:, x].astype(np.float32)
                right_col = warped_right[:, x].astype(np.float32)

                # Only blend where both have content
                left_mask = left_col.sum(axis=1) > 0
                right_mask = right_col.sum(axis=1) > 0
                both = left_mask & right_mask

                blended = left_col.copy()
                blended[both] = (1 - alpha) * left_col[both] + alpha * right_col[both]
                blended[~left_mask & right_mask] = right_col[~left_mask & right_mask]

                canvas[:, x] = blended.astype(np.uint8)

    # Fill remaining right-only region
    right_only_mask = (canvas.sum(axis=2) == 0) & (warped_right.sum(axis=2) > 0)
    canvas[right_only_mask] = warped_right[right_only_mask]

    return canvas


def stitch_videos(left_path: str, right_path: str,
                  output_path: str,
                  cal_path: str = None,
                  cal_date: str = None,
                  frame_offset: int = 0,
                  progress_callback=None) -> str:
    """
    Stitch two video files into a single panoramic video.

    Args:
        left_path: Path to left camera video.
        right_path: Path to right camera video.
        output_path: Path for output stitched video.
        cal_path: Path to existing calibration file (optional).
        cal_date: Date for calibration file naming.
        frame_offset: Frame offset between cameras (right leads by this many frames).
        progress_callback: Optional callback(current_frame, total_frames).

    Returns:
        Path to the output video.
    """
    # Step 1: Calibration
    if cal_path and os.path.exists(cal_path):
        print(f"Loading calibration from {cal_path}")
        cal_data = load_calibration(cal_path)
    else:
        print("Running calibration...")
        if cal_date is None:
            cal_date = date.today().strftime("%Y-%m-%d")
        # Use frame 0 for calibration from videos to be safe with short clips
        cal_data = calibrate(left_path, right_path, cal_date=cal_date,
                             frame_index=0)
        cal_path = os.path.join("calibrations", f"{cal_date}_cal.json")

    # Validate calibration data
    required_keys = ["homography", "canvas_width", "canvas_height",
                     "blend_x_start", "blend_x_end", "offset_x", "offset_y"]
    missing = [k for k in required_keys if k not in cal_data]
    if missing:
        raise ValueError(f"Calibration file missing required keys: {missing}")

    H = np.array(cal_data["homography"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")
    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        raise ValueError("Homography contains NaN or Inf values")

    canvas_w = cal_data["canvas_width"]
    canvas_h = cal_data["canvas_height"]
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError(f"Invalid canvas dimensions: {canvas_w}x{canvas_h}")

    blend_start = cal_data["blend_x_start"]
    blend_end = cal_data["blend_x_end"]
    offset_x = cal_data["offset_x"]
    offset_y = cal_data["offset_y"]

    # Step 2: Open video streams
    cap_left = cv2.VideoCapture(left_path)
    cap_right = cv2.VideoCapture(right_path)

    if not cap_left.isOpened() or not cap_right.isOpened():
        raise FileNotFoundError("Cannot open one or both video files")

    fps = cap_left.get(cv2.CAP_PROP_FPS) or 30.0
    total_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    total_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))

    # Apply frame offset (skip frames on the leading camera)
    if frame_offset > 0:
        # Right started earlier - skip first N frames of right
        for _ in range(frame_offset):
            cap_right.read()
    elif frame_offset < 0:
        # Left started earlier - skip first N frames of left
        for _ in range(-frame_offset):
            cap_left.read()

    total_frames = min(
        total_left - max(0, -frame_offset),
        total_right - max(0, frame_offset)
    )

    # Step 3: Set up output video
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    # Step 4: Process frames
    print(f"Stitching {total_frames} frames at {fps:.1f} fps...")
    print(f"Output: {canvas_w}x{canvas_h}")

    frame_num = 0
    try:
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                if frame_num < total_frames:
                    which = "left" if not ret_left else "right"
                    print(f"  Warning: {which} video ended at frame {frame_num}/{total_frames}")
                break

            stitched = stitch_frame(
                frame_left, frame_right,
                H, canvas_w, canvas_h,
                offset_x, offset_y,
                blend_start, blend_end
            )

            writer.write(stitched)
            frame_num += 1

            if progress_callback:
                progress_callback(frame_num, total_frames)
            elif frame_num % 100 == 0 or frame_num == 1:
                pct = frame_num / max(total_frames, 1) * 100
                print(f"  Frame {frame_num}/{total_frames} ({pct:.1f}%)")
    finally:
        cap_left.release()
        cap_right.release()
        writer.release()

    print(f"Stitching complete: {frame_num} frames written to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Batch panorama stitching")
    parser.add_argument("--left", required=True, help="Left camera video")
    parser.add_argument("--right", required=True, help="Right camera video")
    parser.add_argument("--output", default="output/match_stitched.mp4",
                        help="Output stitched video path")
    parser.add_argument("--cal", default=None, help="Calibration JSON file")
    parser.add_argument("--date", default=None, help="Match date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Auto-detect sync method
    print("Checking for timecode sync...")
    tc_offset = detect_timecode_offset(args.left, args.right)

    if tc_offset is not None:
        print(f"Timecode sync detected: offset = {tc_offset:+.4f}s")
        fps = get_video_info(args.left)["fps"]
        frame_offset = round(tc_offset * fps)
    else:
        print("No timecode found, falling back to audio sync...")
        from sync_audio import sync_audio
        offset_sec = sync_audio(args.left, args.right)
        fps = get_video_info(args.left)["fps"]
        frame_offset = round(offset_sec * fps)

    print(f"Frame offset: {frame_offset} frames")

    stitch_videos(
        args.left, args.right, args.output,
        cal_path=args.cal,
        cal_date=args.date,
        frame_offset=frame_offset
    )


if __name__ == "__main__":
    main()
