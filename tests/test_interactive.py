"""
Tests for interactive.py

Tests run in headless mode (no display window). Verifies:
1. Video opens and frames are read
2. Crop rectangle is positioned correctly
3. Arrow key / zoom controls modify crop state
4. Escape quits cleanly
5. Log is recorded correctly
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interactive import CropState, InteractiveViewer


def make_test_video(tmp_path, width=800, height=400, num_frames=30, fps=30.0):
    """Create a synthetic test video."""
    path = str(tmp_path / "test_pano.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    rng = np.random.RandomState(42)
    for i in range(num_frames):
        frame = rng.randint(50, 200, (height, width, 3), dtype=np.uint8)
        # Add a moving circle to have some visual reference
        cx = int(width * (i / num_frames))
        cv2.circle(frame, (cx, height // 2), 30, (0, 255, 0), -1)
        writer.write(frame)

    writer.release()
    return path


class TestCropState:
    def test_initial_position_is_centered(self):
        crop = CropState(8000, 3000)
        assert crop.center_x == 4000
        assert crop.center_y == 1500

    def test_crop_dimensions_at_1x_zoom(self):
        crop = CropState(8000, 3000)
        crop.zoom = 1.0
        assert crop.crop_w == 1920
        assert crop.crop_h == 1080

    def test_crop_dimensions_at_max_zoom(self):
        crop = CropState(8000, 3000)
        crop.zoom = 1.2
        assert crop.crop_w == 1600
        assert crop.crop_h == 900

    def test_crop_dimensions_at_full_pano(self):
        crop = CropState(8000, 3000)
        crop.zoom = 0
        assert crop.crop_w == 8000
        assert crop.crop_h == 3000

    def test_move_right(self):
        crop = CropState(8000, 3000)
        original_x = crop.center_x
        crop.move(100, 0)
        assert crop.center_x == original_x + 100
        assert crop.center_y == 1500  # Y unchanged

    def test_move_clamped_to_bounds(self):
        crop = CropState(8000, 3000)
        crop.zoom = 1.0
        # Try to move way past the right edge
        crop.move(50000, 0)
        assert crop.crop_x >= 0
        assert crop.crop_x + crop.crop_w <= 8000

    def test_zoom_in(self):
        crop = CropState(8000, 3000)
        crop.zoom = 1.0
        crop.adjust_zoom(0.1)
        assert crop.zoom == pytest.approx(1.1, abs=0.01)
        assert crop.crop_w < 1920  # Crop gets smaller when zoomed in

    def test_zoom_out_to_full_pano(self):
        crop = CropState(8000, 3000)
        crop.zoom = 0.2
        crop.adjust_zoom(-0.2)
        assert crop.zoom == 0.0

    def test_zoom_max_clamped(self):
        crop = CropState(8000, 3000)
        crop.zoom = 1.2
        crop.adjust_zoom(0.5)
        assert crop.zoom == 1.2  # Should not exceed max


class TestInteractiveViewer:
    def test_opens_video_and_reads_metadata(self, tmp_path):
        video_path = make_test_video(tmp_path)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()

        assert viewer.pano_width == 800
        assert viewer.pano_height == 400
        assert viewer.total_frames == 30
        assert abs(viewer.fps - 30.0) < 1.0

        viewer.cap.release()

    def test_crop_state_initialized_on_open(self, tmp_path):
        video_path = make_test_video(tmp_path)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()

        assert viewer.crop is not None
        assert viewer.crop.pano_width == 800
        assert viewer.crop.pano_height == 400

        viewer.cap.release()

    def test_headless_run_processes_all_frames(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=10)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        assert viewer.current_frame == 10
        assert len(viewer.log_rows) == 10

    def test_headless_run_max_frames(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=30)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True, max_frames=5)

        assert viewer.current_frame == 5
        assert len(viewer.log_rows) == 5

    def test_log_has_correct_columns(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=5)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        expected_keys = {
            "frame", "timestamp", "crop_x", "crop_y", "crop_w", "crop_h",
            "home_score", "away_score", "clock_running", "clock_seconds",
            "half", "scoreboard_visible"
        }
        for row in viewer.log_rows:
            assert set(row.keys()) == expected_keys

    def test_log_frame_numbers_increment(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=10)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        for i, row in enumerate(viewer.log_rows):
            assert row["frame"] == i

    def test_save_log_writes_csv(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=5)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        log_path = str(tmp_path / "test_log.csv")
        viewer.save_log(log_path)

        assert os.path.exists(log_path)

        # Read and verify CSV
        import csv
        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 5
        assert rows[0]["frame"] == "0"
        assert "crop_x" in rows[0]

    def test_scoreboard_state_defaults(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=1)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        assert viewer.home_score == 0
        assert viewer.away_score == 0
        assert viewer.half == 1
        assert viewer.scoreboard_visible is True

    def test_clock_ticks_when_running(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=30, fps=30.0)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()

        # Manually start clock and run a few frames in headless
        viewer.clock_running = True
        viewer.run(headless=True, max_frames=30)

        # Clock should have advanced ~1 second (30 frames at 30fps)
        assert viewer.clock_seconds > 0.5
        assert viewer.clock_seconds < 1.5


class TestJoystickConfig:
    def test_default_controller_mapping(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=1)
        viewer = InteractiveViewer(video_path)

        assert viewer.axis_pan == 0
        assert viewer.axis_tilt == 1
        assert viewer.axis_zoom_in == 5
        assert viewer.axis_zoom_out == 4
        assert viewer.button_snap_center == 1
        assert viewer.button_snap_left == 0
        assert viewer.button_snap_right == 3
        assert viewer.button_wide_view == 2
        assert viewer.deadzone == 0.10

    def test_custom_controller_mapping(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=1)
        config = {
            "controller": {
                "axis_pan": 2,
                "axis_tilt": 3,
                "deadzone": 0.15,
            }
        }
        viewer = InteractiveViewer(video_path, config=config)
        assert viewer.axis_pan == 2
        assert viewer.axis_tilt == 3
        assert viewer.deadzone == 0.15
        # Defaults for unspecified
        assert viewer.axis_zoom_in == 5

    def test_snap_positions_from_config(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=1)
        config = {
            "snap_center_x": 4000,
            "snap_left_goal_x": 500,
            "snap_right_goal_x": 7500,
            "snap_y": 1494,
        }
        viewer = InteractiveViewer(video_path, config=config)
        assert viewer.snap_center_x == 4000
        assert viewer.snap_left_goal_x == 500
        assert viewer.snap_right_goal_x == 7500
        assert viewer.snap_y == 1494

    def test_joystick_handle_without_controller(self, tmp_path):
        """handle_joystick should not crash when no controller is connected."""
        video_path = make_test_video(tmp_path, num_frames=1)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()
        # joystick is None, should not raise
        viewer.handle_joystick()
        viewer.cap.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
