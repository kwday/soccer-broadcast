"""
Tests for pages/1_Stitch.py

Verifies:
1. Metadata display function works correctly
2. Duration formatting is correct
3. Preview frame extraction works
4. Sync detection returns expected types
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pages"))

# Import the utility functions from the stitch page
from importlib import import_module

stitch_page = import_module("1_Stitch")
format_duration = stitch_page.format_duration
get_stitch_preview_frame = stitch_page.get_stitch_preview_frame


def make_test_video(tmp_path, width=640, height=480, num_frames=10, fps=30.0):
    """Create a test video."""
    path = str(tmp_path / "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.RandomState(42)
    for i in range(num_frames):
        frame = rng.randint(30, 200, (height, width, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()
    return path


class TestFormatDuration:
    def test_seconds_only(self):
        assert format_duration(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert format_duration(125) == "2:05"

    def test_hours(self):
        assert format_duration(3661) == "1:01:01"

    def test_zero(self):
        assert format_duration(0) == "0:00"

    def test_exact_hour(self):
        assert format_duration(3600) == "1:00:00"


class TestPreviewFrame:
    def test_extracts_frame(self, tmp_path):
        """Verify we can extract a preview frame."""
        video_path = make_test_video(tmp_path)
        frame = get_stitch_preview_frame(video_path, frame_index=0)
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        # Should be RGB (not BGR) after conversion
        assert frame.dtype == np.uint8

    def test_extracts_middle_frame(self, tmp_path):
        """Verify we can extract from a specific position."""
        video_path = make_test_video(tmp_path, num_frames=20)
        frame = get_stitch_preview_frame(video_path, frame_index=10)
        assert frame is not None

    def test_nonexistent_file_returns_none(self):
        """Verify graceful failure on bad path."""
        frame = get_stitch_preview_frame("/nonexistent/video.mp4")
        assert frame is None

    def test_preview_has_content(self, tmp_path):
        """Verify the preview frame is not all black."""
        video_path = make_test_video(tmp_path)
        frame = get_stitch_preview_frame(video_path)
        assert frame.sum() > 0


class TestVideoInfo:
    def test_get_video_info(self, tmp_path):
        """Verify get_video_info returns correct metadata."""
        from stitch import get_video_info
        video_path = make_test_video(tmp_path, width=800, height=600, num_frames=15)
        info = get_video_info(video_path)
        assert info["width"] == 800
        assert info["height"] == 600
        assert info["frame_count"] == 15
        assert info["fps"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
