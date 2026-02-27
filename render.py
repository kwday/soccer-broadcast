"""
render.py — Final broadcast encoder.

Reads the log CSV from an interactive session, decodes the stitched panoramic
video, crops each frame according to the log, composites the scoreboard overlay,
and encodes the final 1080p broadcast output.

Usage:
    python render.py --video stitched.mp4 --log logs/match.csv --output output/broadcast.mp4
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

from scoreboard import ScoreboardRenderer, ScoreboardState


def read_log(log_path: str) -> list:
    """Read the session log CSV and return a list of row dicts."""
    with open(log_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def render_broadcast(video_path: str, log_path: str, output_path: str,
                     home_team: str = "HOME", away_team: str = "AWAY",
                     home_color: str = "#1E5E3A", away_color: str = "#5E1E1E",
                     output_width: int = 1920, output_height: int = 1080,
                     output_fps: float = 30.0, output_crf: int = 18,
                     font_path: str = None, mono_font_path: str = None,
                     progress_callback=None) -> str:
    """
    Render the final broadcast video from a stitched panorama and session log.

    Args:
        video_path: Path to the stitched panoramic video.
        log_path: Path to the session log CSV.
        output_path: Path for the output broadcast video.
        home_team: Home team name.
        away_team: Away team name.
        home_color: Home team color (hex).
        away_color: Away team color (hex).
        output_width: Output video width (default 1920).
        output_height: Output video height (default 1080).
        output_fps: Output frame rate.
        output_crf: CRF quality (not used with OpenCV VideoWriter).
        font_path: Path to font file for scoreboard.
        mono_font_path: Path to monospace font for timer.
        progress_callback: Optional callback(current_frame, total_frames).

    Returns:
        Path to the output video.
    """
    # Read log
    print(f"Reading log: {log_path}")
    log_rows = read_log(log_path)
    total_frames = len(log_rows)
    print(f"  {total_frames} frames to render")

    # Open source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    pano_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pano_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"  Source: {pano_width}x{pano_height} @ {source_fps:.1f}fps")

    # Initialize scoreboard renderer
    renderer = ScoreboardRenderer(
        width=output_width, height=output_height,
        font_path=font_path, mono_font_path=mono_font_path
    )

    # Set up output video
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps,
                             (output_width, output_height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    print(f"Rendering to {output_path} ({output_width}x{output_height} @ {output_fps}fps)...")

    frame_num = 0
    for row in log_rows:
        # Read source frame
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: source video ended at frame {frame_num}")
            break

        # Parse crop parameters
        crop_x = int(row["crop_x"])
        crop_y = int(row["crop_y"])
        crop_w = int(row["crop_w"])
        crop_h = int(row["crop_h"])

        # Clamp to source bounds
        crop_x = max(0, min(crop_x, pano_width - crop_w))
        crop_y = max(0, min(crop_y, pano_height - crop_h))
        crop_w = min(crop_w, pano_width - crop_x)
        crop_h = min(crop_h, pano_height - crop_y)

        # Crop
        cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        # Handle letterboxing for full panorama view (zoom=0)
        if crop_w == pano_width and crop_h == pano_height:
            # Scale to fit width, letterbox vertically
            scale = output_width / crop_w
            scaled_h = int(crop_h * scale)
            scaled = cv2.resize(cropped, (output_width, scaled_h))

            # Center in output frame with black bars
            output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            y_offset = (output_height - scaled_h) // 2
            output_frame[y_offset:y_offset + scaled_h] = scaled
        else:
            # Normal crop: resize to output dimensions
            output_frame = cv2.resize(cropped, (output_width, output_height))

        # Composite scoreboard
        scoreboard_visible = row.get("scoreboard_visible", "true") == "true"
        state = ScoreboardState(
            home_team=home_team,
            away_team=away_team,
            home_score=int(row.get("home_score", 0)),
            away_score=int(row.get("away_score", 0)),
            clock_seconds=int(row.get("clock_seconds", 0)),
            half=int(row.get("half", 1)),
            visible=scoreboard_visible,
            home_color=home_color,
            away_color=away_color,
        )
        output_frame = renderer.composite_onto_frame(output_frame, state)

        # Write frame
        writer.write(output_frame)
        frame_num += 1

        if progress_callback:
            progress_callback(frame_num, total_frames)
        elif frame_num % 100 == 0 or frame_num == 1:
            pct = frame_num / max(total_frames, 1) * 100
            print(f"  Frame {frame_num}/{total_frames} ({pct:.1f}%)")

    cap.release()
    writer.release()

    print(f"Render complete: {frame_num} frames written to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Final broadcast encoder")
    parser.add_argument("--video", required=True, help="Stitched panoramic video")
    parser.add_argument("--log", required=True, help="Session log CSV")
    parser.add_argument("--output", default="output/match_broadcast_1080p.mp4",
                        help="Output broadcast video path")
    parser.add_argument("--home-team", default="HOME")
    parser.add_argument("--away-team", default="AWAY")
    parser.add_argument("--home-color", default="#1E5E3A")
    parser.add_argument("--away-color", default="#5E1E1E")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    render_broadcast(
        args.video, args.log, args.output,
        home_team=args.home_team, away_team=args.away_team,
        home_color=args.home_color, away_color=args.away_color,
        output_width=args.width, output_height=args.height,
        output_fps=args.fps
    )


if __name__ == "__main__":
    main()
