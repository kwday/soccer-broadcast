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
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    with open(log_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Log file is empty: {log_path}")
    required = ["crop_x", "crop_y", "crop_w", "crop_h"]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"Log file missing required columns: {missing}")
    return rows


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
    try:
        for row in log_rows:
            # Read source frame
            ret, frame = cap.read()
            if not ret:
                print(f"  Warning: source video ended at frame {frame_num}/{total_frames}")
                break

            # Parse crop parameters with validation
            try:
                crop_x = int(row["crop_x"])
                crop_y = int(row["crop_y"])
                crop_w = int(row["crop_w"])
                crop_h = int(row["crop_h"])
            except (ValueError, KeyError) as e:
                print(f"  Warning: bad crop data at frame {frame_num}: {e}, skipping")
                frame_num += 1
                continue

            # Clamp to source bounds
            crop_x = max(0, min(crop_x, pano_width - crop_w))
            crop_y = max(0, min(crop_y, pano_height - crop_h))
            crop_w = min(crop_w, pano_width - crop_x)
            crop_h = min(crop_h, pano_height - crop_y)

            # Validate crop dimensions
            if crop_w <= 0 or crop_h <= 0:
                print(f"  Warning: invalid crop at frame {frame_num}, using full frame")
                crop_x, crop_y = 0, 0
                crop_w, crop_h = pano_width, pano_height

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
            try:
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
            except (ValueError, Exception) as e:
                # If scoreboard fails, write frame without it
                if frame_num == 0:
                    print(f"  Warning: scoreboard error: {e}")

            # Write frame
            writer.write(output_frame)
            frame_num += 1

            if progress_callback:
                progress_callback(frame_num, total_frames)
            elif frame_num % 100 == 0 or frame_num == 1:
                pct = frame_num / max(total_frames, 1) * 100
                print(f"  Frame {frame_num}/{total_frames} ({pct:.1f}%)")
    finally:
        cap.release()
        writer.release()

    print(f"Render complete: {frame_num} frames written to {output_path}")
    return output_path


def _get_ffmpeg_path() -> str:
    """Find a working ffmpeg executable."""
    import shutil

    # Try imageio-ffmpeg first (standalone binary, most reliable)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    # Try system ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    return "ffmpeg"  # Fallback, let it fail with a clear error


def _get_ffprobe_path() -> str:
    """Find a working ffprobe executable."""
    import shutil

    # Try imageio-ffmpeg directory (ffprobe is usually alongside ffmpeg)
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        ffprobe = ffmpeg_path.replace("ffmpeg", "ffprobe")
        if os.path.exists(ffprobe):
            return ffprobe
    except ImportError:
        pass

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe

    return "ffprobe"


def mux_audio(video_path: str, audio_source: str, output_path: str) -> str:
    """
    Mux audio from a source file into the rendered video using ffmpeg.

    Args:
        video_path: Path to the rendered video (no audio).
        audio_source: Path to audio source (typically left camera file).
        output_path: Path for the final output with audio.

    Returns:
        Path to the output file.
    """
    import subprocess
    import tempfile

    print(f"Muxing audio from {audio_source}...")

    ffmpeg = _get_ffmpeg_path()

    # Use a temp file for the intermediate step
    temp_output = output_path + ".tmp.mp4"

    cmd = [
        ffmpeg, "-y",
        "-i", video_path,      # video stream
        "-i", audio_source,    # audio source
        "-c:v", "libx264",     # re-encode video as H.264 for compatibility
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",         # encode audio as AAC
        "-b:a", "192k",
        "-map", "0:v:0",       # take video from first input
        "-map", "1:a:0",       # take audio from second input
        "-shortest",           # stop when shorter stream ends
        temp_output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "(no output)"
        print(f"  Warning: ffmpeg mux failed (code {result.returncode}): {error_msg[:500]}")
        # If mux fails, just return the video-only file
        return video_path

    # Replace original with muxed version
    if os.path.exists(temp_output):
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_output, output_path)
        print(f"Audio muxed: {output_path}")
    else:
        print("  Warning: muxed file not created, using video-only output")
        return video_path

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
    parser.add_argument("--audio", default=None,
                        help="Audio source file (e.g., left camera video)")
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

    if args.audio:
        mux_audio(args.output, args.audio, args.output)


if __name__ == "__main__":
    main()
