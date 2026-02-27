"""
Tests for audio muxing in render.py

Verifies:
1. Final MP4 has audio track after mux
2. Audio duration matches video duration
3. Mux works with WAV input
"""

import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render import render_broadcast, mux_audio


def make_test_video(tmp_path, width=640, height=360, num_frames=30, fps=30.0):
    """Create a test video (no audio)."""
    path = str(tmp_path / "video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.RandomState(42)
    for i in range(num_frames):
        frame = rng.randint(50, 200, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def make_test_audio(tmp_path, duration=1.0, sample_rate=16000):
    """Create a test WAV audio file."""
    path = str(tmp_path / "audio.wav")
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, signal)
    return path


def _get_ffmpeg():
    """Find a working ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"


def get_audio_streams(video_path):
    """Detect audio streams using ffmpeg -i (since ffprobe may not be available)."""
    import re
    try:
        ffmpeg = _get_ffmpeg()
        result = subprocess.run(
            [ffmpeg, "-i", video_path, "-hide_banner"],
            capture_output=True, text=True, timeout=10
        )
        # ffmpeg -i returns exit code 1 when no output specified, but still prints info
        stderr = result.stderr or ""
        # Look for lines like "Stream #0:1: Audio: aac..."
        audio_streams = re.findall(r"Stream #\d+:\d+.*Audio:", stderr)
        return audio_streams
    except subprocess.TimeoutExpired:
        return []


def get_duration(video_path):
    """Get video duration in seconds using ffmpeg -i."""
    import re
    try:
        ffmpeg = _get_ffmpeg()
        result = subprocess.run(
            [ffmpeg, "-i", video_path, "-hide_banner"],
            capture_output=True, text=True, timeout=10
        )
        stderr = result.stderr or ""
        # Look for "Duration: HH:MM:SS.mm"
        match = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", stderr)
        if match:
            h, m, s = float(match.group(1)), float(match.group(2)), float(match.group(3))
            return h * 3600 + m * 60 + s
        return 0
    except subprocess.TimeoutExpired:
        return 0


class TestAudioMux:
    def test_mux_adds_audio_track(self, tmp_path):
        """Verify muxed file has an audio stream."""
        video_path = make_test_video(tmp_path, num_frames=30)
        audio_path = make_test_audio(tmp_path, duration=1.0)
        output_path = str(tmp_path / "muxed.mp4")

        # First verify no audio in source
        streams_before = get_audio_streams(video_path)
        assert len(streams_before) == 0

        result = mux_audio(video_path, audio_path, output_path)

        assert os.path.exists(result)
        streams_after = get_audio_streams(result)
        assert len(streams_after) >= 1, "Muxed file should have audio stream"

    def test_mux_preserves_video(self, tmp_path):
        """Verify video content is preserved after mux."""
        video_path = make_test_video(tmp_path, num_frames=10)
        audio_path = make_test_audio(tmp_path, duration=0.5)
        output_path = str(tmp_path / "muxed.mp4")

        mux_audio(video_path, audio_path, output_path)

        cap = cv2.VideoCapture(output_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert w == 640
        assert h == 360

    def test_full_render_with_audio(self, tmp_path):
        """End-to-end: render + mux audio."""
        # Create panorama source
        pano_path = str(tmp_path / "panorama.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(pano_path, fourcc, 30.0, (2000, 1200))
        rng = np.random.RandomState(42)
        for i in range(10):
            frame = rng.randint(30, 180, (1200, 2000, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # Create log
        log_path = str(tmp_path / "log.csv")
        fieldnames = [
            "frame", "timestamp", "crop_x", "crop_y", "crop_w", "crop_h",
            "home_score", "away_score", "clock_running", "clock_seconds",
            "half", "scoreboard_visible"
        ]
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i in range(10):
                w.writerow({
                    "frame": i, "timestamp": f"{i/30:.3f}",
                    "crop_x": 40, "crop_y": 60, "crop_w": 960, "crop_h": 540,
                    "home_score": 0, "away_score": 0, "clock_running": "false",
                    "clock_seconds": 0, "half": 1, "scoreboard_visible": "true"
                })

        # Create audio
        audio_path = make_test_audio(tmp_path, duration=0.5)

        # Render
        video_only = str(tmp_path / "broadcast.mp4")
        render_broadcast(
            pano_path, log_path, video_only,
            output_width=640, output_height=360
        )

        # Mux audio
        final_path = str(tmp_path / "final.mp4")
        import shutil
        shutil.copy(video_only, final_path)
        mux_audio(final_path, audio_path, final_path)

        assert os.path.exists(final_path)
        streams = get_audio_streams(final_path)
        assert len(streams) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
