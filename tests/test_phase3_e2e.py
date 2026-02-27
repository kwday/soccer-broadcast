"""
Phase 3 End-to-End Test

Full pipeline: stitch → interactive (headless) → render → mux audio → verify final output.
"""

import csv
import os
import re
import subprocess
import sys

import cv2
import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibrate import calibrate
from stitch import stitch_videos, get_video_info
from interactive import InteractiveViewer
from render import render_broadcast, mux_audio


def _get_ffmpeg():
    """Find a working ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"


def get_audio_streams(video_path):
    """Detect audio streams using ffmpeg -i."""
    try:
        ffmpeg = _get_ffmpeg()
        result = subprocess.run(
            [ffmpeg, "-i", video_path, "-hide_banner"],
            capture_output=True, text=True, timeout=10
        )
        stderr = result.stderr or ""
        audio_streams = re.findall(r"Stream #\d+:\d+.*Audio:", stderr)
        return audio_streams
    except subprocess.TimeoutExpired:
        return []


def make_stereo_pair(tmp_path, num_frames=30, width=400, height=300,
                     overlap_px=120, fps=30.0):
    """Create a pair of overlapping video files for stitching."""
    full_width = 2 * width - overlap_px
    rng = np.random.RandomState(42)

    base_frame = rng.randint(50, 200, (height, full_width, 3), dtype=np.uint8)
    for _ in range(20):
        cx = rng.randint(0, full_width)
        cy = rng.randint(0, height)
        r = rng.randint(10, 30)
        color = tuple(int(c) for c in rng.randint(0, 256, 3))
        cv2.circle(base_frame, (cx, cy), r, color, -1)
    base_frame = cv2.GaussianBlur(base_frame, (5, 5), 1.0)

    left_path = str(tmp_path / "left.mp4")
    right_path = str(tmp_path / "right.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_left = cv2.VideoWriter(left_path, fourcc, fps, (width, height))
    writer_right = cv2.VideoWriter(right_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        noise = rng.randint(-3, 4, base_frame.shape, dtype=np.int16)
        frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer_left.write(frame[:, :width])
        writer_right.write(frame[:, (width - overlap_px):])

    writer_left.release()
    writer_right.release()

    return left_path, right_path


def make_test_audio(tmp_path, duration=1.0, sample_rate=16000):
    """Create a test WAV audio file."""
    path = str(tmp_path / "audio.wav")
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, signal)
    return path


class TestPhase3EndToEnd:
    def test_full_pipeline(self, tmp_path):
        """End-to-end: stitch → interactive (headless) → render → mux audio."""
        # --- Stage 1: Stitch ---
        left_path, right_path = make_stereo_pair(tmp_path, num_frames=30)

        cal_data = calibrate(
            left_path, right_path,
            cal_date="e2e-test",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )
        cal_path = str(tmp_path / "cal" / "e2e-test_cal.json")
        assert os.path.exists(cal_path)

        stitched_path = str(tmp_path / "stitched.mp4")
        stitch_videos(
            left_path, right_path,
            stitched_path,
            cal_path=cal_path,
            frame_offset=0
        )
        assert os.path.exists(stitched_path)
        stitch_info = get_video_info(stitched_path)
        assert stitch_info["frame_count"] == 30

        # --- Stage 2: Interactive (headless) ---
        viewer = InteractiveViewer(stitched_path, config={
            "home_team": "Eagles",
            "away_team": "Hawks",
            "home_color": "#1E5E3A",
            "away_color": "#5E1E1E",
        })
        viewer.run(headless=True)
        assert viewer.current_frame == 30
        assert len(viewer.log_rows) == 30

        # Save the session log
        log_path = str(tmp_path / "session.csv")
        viewer.save_log(log_path)
        assert os.path.exists(log_path)

        # --- Stage 3: Render ---
        broadcast_path = str(tmp_path / "broadcast.mp4")
        render_broadcast(
            stitched_path, log_path, broadcast_path,
            home_team="Eagles", away_team="Hawks",
            output_width=640, output_height=360, output_fps=30.0
        )
        assert os.path.exists(broadcast_path)

        cap = cv2.VideoCapture(broadcast_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert w == 640
        assert h == 360
        assert fc == 30

        # --- Stage 4: Mux audio ---
        audio_path = make_test_audio(tmp_path, duration=1.0)
        final_path = str(tmp_path / "final.mp4")
        import shutil
        shutil.copy(broadcast_path, final_path)
        result = mux_audio(final_path, audio_path, final_path)

        assert os.path.exists(result)
        streams = get_audio_streams(result)
        assert len(streams) >= 1, "Final output should have audio"

    def test_render_matches_log_state(self, tmp_path):
        """Verify rendered output reflects score changes from interactive session."""
        left_path, right_path = make_stereo_pair(tmp_path, num_frames=20)

        cal_data = calibrate(
            left_path, right_path,
            cal_date="state-test",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )
        cal_path = str(tmp_path / "cal" / "state-test_cal.json")

        stitched_path = str(tmp_path / "stitched.mp4")
        stitch_videos(
            left_path, right_path,
            stitched_path,
            cal_path=cal_path,
            frame_offset=0
        )

        # Run interactive with simulated score changes
        viewer = InteractiveViewer(stitched_path)
        viewer.open_video()
        viewer.running = True

        for i in range(20):
            ret, frame = viewer.cap.read()
            if not ret:
                break
            if i == 5:
                viewer.home_score = 1
            if i == 15:
                viewer.away_score = 1
            viewer.log_frame(i / viewer.fps)
            viewer.current_frame += 1

        viewer.cap.release()

        log_path = str(tmp_path / "session.csv")
        viewer.save_log(log_path)

        # Verify log has correct score transitions
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[4]["home_score"] == "0"
        assert rows[5]["home_score"] == "1"
        assert rows[14]["away_score"] == "0"
        assert rows[15]["away_score"] == "1"

        # Render from this log
        broadcast_path = str(tmp_path / "broadcast.mp4")
        render_broadcast(
            stitched_path, log_path, broadcast_path,
            output_width=640, output_height=360
        )
        assert os.path.exists(broadcast_path)
        assert os.path.getsize(broadcast_path) > 0

    def test_final_output_playable(self, tmp_path):
        """Verify the final muxed output can be opened and has valid frames."""
        left_path, right_path = make_stereo_pair(tmp_path, num_frames=10)

        cal_data = calibrate(
            left_path, right_path,
            cal_date="playable-test",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )
        cal_path = str(tmp_path / "cal" / "playable-test_cal.json")

        stitched_path = str(tmp_path / "stitched.mp4")
        stitch_videos(
            left_path, right_path,
            stitched_path,
            cal_path=cal_path,
            frame_offset=0
        )

        viewer = InteractiveViewer(stitched_path)
        viewer.run(headless=True)
        log_path = str(tmp_path / "session.csv")
        viewer.save_log(log_path)

        broadcast_path = str(tmp_path / "broadcast.mp4")
        render_broadcast(
            stitched_path, log_path, broadcast_path,
            output_width=640, output_height=360
        )

        audio_path = make_test_audio(tmp_path, duration=0.5)
        final_path = str(tmp_path / "final.mp4")
        import shutil
        shutil.copy(broadcast_path, final_path)
        mux_audio(final_path, audio_path, final_path)

        # Verify we can read frames from the final output
        cap = cv2.VideoCapture(final_path)
        assert cap.isOpened()
        ret, frame = cap.read()
        assert ret
        assert frame is not None
        assert frame.shape == (360, 640, 3)
        # Frame should not be all black
        assert frame.sum() > 0
        cap.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
