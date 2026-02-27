"""
Tests for stitch.py

Creates short synthetic video pairs with known overlap and verifies:
1. Output video exists
2. Resolution matches expected panorama dimensions
3. Frame count is correct
4. Stitch seam blends correctly (no hard edges)
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stitch import stitch_frame, stitch_videos, load_calibration, get_video_info
from calibrate import calibrate


def make_test_video_pair(tmp_path, num_frames=10, width=400, height=300,
                         overlap_px=120, fps=30.0):
    """
    Create two short synthetic video files with known overlap.
    Returns left_path, right_path, and expected panorama width.
    """
    full_width = 2 * width - overlap_px
    rng = np.random.RandomState(42)

    # Create a base frame with distinctive features
    base_frame = rng.randint(50, 200, (height, full_width, 3), dtype=np.uint8)

    # Add shapes for visual features
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
        # Add slight variation per frame (simulates camera movement)
        noise = rng.randint(-5, 6, base_frame.shape, dtype=np.int16)
        frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        left_frame = frame[:, :width]
        right_frame = frame[:, (width - overlap_px):]

        writer_left.write(left_frame)
        writer_right.write(right_frame)

    writer_left.release()
    writer_right.release()

    return left_path, right_path, full_width


class TestGetVideoInfo:
    def test_reads_video_metadata(self, tmp_path):
        path = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
        for _ in range(5):
            w.write(np.zeros((480, 640, 3), dtype=np.uint8))
        w.release()

        info = get_video_info(path)
        assert info["width"] == 640
        assert info["height"] == 480
        assert abs(info["fps"] - 30.0) < 1.0
        assert info["frame_count"] == 5


class TestStitchFrame:
    def test_stitched_frame_has_correct_dimensions(self, tmp_path):
        """Verify a single frame stitch produces correct canvas size."""
        left_path, right_path, _ = make_test_video_pair(tmp_path, num_frames=2)

        # Run calibration
        cal_data = calibrate(
            left_path, right_path,
            cal_date="test",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )

        H = np.array(cal_data["homography"])
        canvas_w = cal_data["canvas_width"]
        canvas_h = cal_data["canvas_height"]

        # Read one frame from each
        cap_l = cv2.VideoCapture(left_path)
        cap_r = cv2.VideoCapture(right_path)
        _, frame_l = cap_l.read()
        _, frame_r = cap_r.read()
        cap_l.release()
        cap_r.release()

        result = stitch_frame(
            frame_l, frame_r, H, canvas_w, canvas_h,
            cal_data["offset_x"], cal_data["offset_y"],
            cal_data["blend_x_start"], cal_data["blend_x_end"]
        )

        assert result.shape == (canvas_h, canvas_w, 3)
        # Canvas should be wider than either input
        assert canvas_w > 400

    def test_no_black_seam_in_blend_region(self, tmp_path):
        """Verify the blend region doesn't have hard black edges."""
        left_path, right_path, _ = make_test_video_pair(tmp_path, num_frames=2)

        cal_data = calibrate(
            left_path, right_path,
            cal_date="test",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )

        H = np.array(cal_data["homography"])

        cap_l = cv2.VideoCapture(left_path)
        cap_r = cv2.VideoCapture(right_path)
        _, frame_l = cap_l.read()
        _, frame_r = cap_r.read()
        cap_l.release()
        cap_r.release()

        result = stitch_frame(
            frame_l, frame_r, H,
            cal_data["canvas_width"], cal_data["canvas_height"],
            cal_data["offset_x"], cal_data["offset_y"],
            cal_data["blend_x_start"], cal_data["blend_x_end"]
        )

        # Check the center of the blend region for non-black pixels
        blend_mid = (cal_data["blend_x_start"] + cal_data["blend_x_end"]) // 2
        center_y = result.shape[0] // 2
        # Sample a column in the blend region
        col = result[center_y - 20:center_y + 20, blend_mid]
        # At least some pixels should be non-black
        assert col.sum() > 0, "Blend region center is all black"


class TestStitchVideos:
    def test_output_video_exists(self, tmp_path):
        left_path, right_path, _ = make_test_video_pair(tmp_path, num_frames=10)
        output_path = str(tmp_path / "stitched.mp4")

        stitch_videos(
            left_path, right_path, output_path,
            cal_date="test-stitch",
            frame_offset=0
        )

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_output_resolution_matches_calibration(self, tmp_path):
        left_path, right_path, _ = make_test_video_pair(tmp_path, num_frames=10)
        output_path = str(tmp_path / "stitched.mp4")

        stitch_videos(
            left_path, right_path, output_path,
            cal_date="test-res",
            frame_offset=0
        )

        info = get_video_info(output_path)
        # Load the calibration that was auto-created
        cal_path = os.path.join("calibrations", "test-res_cal.json")
        cal_data = load_calibration(cal_path)

        assert info["width"] == cal_data["canvas_width"]
        assert info["height"] == cal_data["canvas_height"]

    def test_frame_count_matches_input(self, tmp_path):
        num_frames = 10
        left_path, right_path, _ = make_test_video_pair(
            tmp_path, num_frames=num_frames
        )
        output_path = str(tmp_path / "stitched.mp4")

        stitch_videos(
            left_path, right_path, output_path,
            cal_date="test-count",
            frame_offset=0
        )

        info = get_video_info(output_path)
        assert info["frame_count"] == num_frames

    def test_with_existing_calibration(self, tmp_path):
        left_path, right_path, _ = make_test_video_pair(tmp_path, num_frames=5)

        # Pre-calibrate
        cal_data = calibrate(
            left_path, right_path,
            cal_date="preexist",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )
        cal_path = str(tmp_path / "cal" / "preexist_cal.json")

        output_path = str(tmp_path / "stitched.mp4")
        stitch_videos(
            left_path, right_path, output_path,
            cal_path=cal_path,
            frame_offset=0
        )

        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["frame_count"] == 5

    def test_progress_callback(self, tmp_path):
        left_path, right_path, _ = make_test_video_pair(tmp_path, num_frames=5)
        output_path = str(tmp_path / "stitched.mp4")

        progress_log = []

        def on_progress(current, total):
            progress_log.append((current, total))

        stitch_videos(
            left_path, right_path, output_path,
            cal_date="test-progress",
            frame_offset=0,
            progress_callback=on_progress
        )

        assert len(progress_log) == 5
        assert progress_log[-1][0] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
