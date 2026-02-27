"""
Tests for error handling across the pipeline.

Verifies:
1. Joystick disconnect recovery
2. End-of-video handling
3. Missing files produce helpful errors
4. Invalid calibration data is caught
5. Malformed CSV data doesn't crash render
"""

import csv
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render import render_broadcast, read_log
from stitch import stitch_videos


def make_test_video(tmp_path, width=640, height=360, num_frames=10, fps=30.0):
    """Create a test video."""
    path = str(tmp_path / "video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.RandomState(42)
    for _ in range(num_frames):
        writer.write(rng.randint(30, 200, (height, width, 3), dtype=np.uint8))
    writer.release()
    return path


class TestMissingFiles:
    def test_render_missing_video(self, tmp_path):
        """Render with missing video raises FileNotFoundError."""
        log_path = str(tmp_path / "log.csv")
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "crop_x", "crop_y", "crop_w", "crop_h"])
            w.writeheader()
            w.writerow({"frame": 0, "crop_x": 0, "crop_y": 0, "crop_w": 320, "crop_h": 180})

        with pytest.raises(FileNotFoundError):
            render_broadcast("/nonexistent/video.mp4", log_path,
                             str(tmp_path / "out.mp4"), output_width=320, output_height=180)

    def test_render_missing_log(self, tmp_path):
        """Render with missing log raises FileNotFoundError."""
        video_path = make_test_video(tmp_path)
        with pytest.raises(FileNotFoundError):
            render_broadcast(video_path, "/nonexistent/log.csv",
                             str(tmp_path / "out.mp4"))

    def test_read_log_missing_file(self):
        """read_log with nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_log("/nonexistent/log.csv")

    def test_read_log_empty_file(self, tmp_path):
        """read_log with empty CSV raises ValueError."""
        path = str(tmp_path / "empty.csv")
        with open(path, "w") as f:
            f.write("frame,crop_x,crop_y,crop_w,crop_h\n")
        with pytest.raises(ValueError, match="empty"):
            read_log(path)

    def test_read_log_missing_columns(self, tmp_path):
        """read_log with missing columns raises ValueError."""
        path = str(tmp_path / "bad.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "something"])
            w.writeheader()
            w.writerow({"frame": 0, "something": "x"})
        with pytest.raises(ValueError, match="missing required columns"):
            read_log(path)


class TestInvalidCalibration:
    def test_missing_keys(self, tmp_path):
        """Stitch with calibration missing keys raises ValueError."""
        cal_path = str(tmp_path / "bad_cal.json")
        with open(cal_path, "w") as f:
            json.dump({"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, f)

        left_path = make_test_video(tmp_path)
        right_path = make_test_video(tmp_path)
        # Rename to avoid collision
        import shutil
        right_path2 = str(tmp_path / "right.mp4")
        shutil.copy(left_path, right_path2)

        with pytest.raises(ValueError, match="missing required keys"):
            stitch_videos(left_path, right_path2,
                          str(tmp_path / "out.mp4"), cal_path=cal_path)

    def test_nan_homography(self, tmp_path):
        """Stitch with NaN homography raises ValueError."""
        cal_path = str(tmp_path / "nan_cal.json")
        cal_data = {
            "homography": [[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]],
            "canvas_width": 800, "canvas_height": 600,
            "blend_x_start": 200, "blend_x_end": 400,
            "offset_x": 0, "offset_y": 0,
        }
        with open(cal_path, "w") as f:
            json.dump(cal_data, f)

        left_path = make_test_video(tmp_path)
        import shutil
        right_path = str(tmp_path / "right.mp4")
        shutil.copy(left_path, right_path)

        with pytest.raises(ValueError, match="NaN"):
            stitch_videos(left_path, right_path,
                          str(tmp_path / "out.mp4"), cal_path=cal_path)

    def test_invalid_canvas_dimensions(self, tmp_path):
        """Stitch with zero canvas dimensions raises ValueError."""
        cal_path = str(tmp_path / "zero_cal.json")
        cal_data = {
            "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "canvas_width": 0, "canvas_height": 600,
            "blend_x_start": 0, "blend_x_end": 0,
            "offset_x": 0, "offset_y": 0,
        }
        with open(cal_path, "w") as f:
            json.dump(cal_data, f)

        left_path = make_test_video(tmp_path)
        import shutil
        right_path = str(tmp_path / "right.mp4")
        shutil.copy(left_path, right_path)

        with pytest.raises(ValueError, match="Invalid canvas"):
            stitch_videos(left_path, right_path,
                          str(tmp_path / "out.mp4"), cal_path=cal_path)


class TestMalformedCSV:
    def test_bad_crop_values_dont_crash(self, tmp_path):
        """Render continues on bad crop values, skipping bad frames."""
        video_path = make_test_video(tmp_path, num_frames=5)
        log_path = str(tmp_path / "log.csv")
        fieldnames = ["frame", "timestamp", "crop_x", "crop_y", "crop_w", "crop_h",
                       "home_score", "away_score", "clock_running", "clock_seconds",
                       "half", "scoreboard_visible"]
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            # Good row
            w.writerow({"frame": 0, "timestamp": "0.000",
                        "crop_x": 0, "crop_y": 0, "crop_w": 320, "crop_h": 180,
                        "home_score": 0, "away_score": 0, "clock_running": "false",
                        "clock_seconds": 0, "half": 1, "scoreboard_visible": "true"})
            # Bad row with non-numeric crop
            w.writerow({"frame": 1, "timestamp": "0.033",
                        "crop_x": "abc", "crop_y": 0, "crop_w": 320, "crop_h": 180,
                        "home_score": 0, "away_score": 0, "clock_running": "false",
                        "clock_seconds": 0, "half": 1, "scoreboard_visible": "true"})
            # Good row
            w.writerow({"frame": 2, "timestamp": "0.066",
                        "crop_x": 0, "crop_y": 0, "crop_w": 320, "crop_h": 180,
                        "home_score": 0, "away_score": 0, "clock_running": "false",
                        "clock_seconds": 0, "half": 1, "scoreboard_visible": "true"})

        output_path = str(tmp_path / "out.mp4")
        # Should not crash — skips the bad frame
        render_broadcast(video_path, log_path, output_path,
                         output_width=320, output_height=180)
        assert os.path.exists(output_path)


class TestEndOfVideo:
    def test_video_shorter_than_log(self, tmp_path):
        """Render handles video ending before log rows exhausted."""
        video_path = make_test_video(tmp_path, num_frames=3)
        log_path = str(tmp_path / "log.csv")
        fieldnames = ["frame", "timestamp", "crop_x", "crop_y", "crop_w", "crop_h",
                       "home_score", "away_score", "clock_running", "clock_seconds",
                       "half", "scoreboard_visible"]
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i in range(10):  # More rows than video frames
                w.writerow({"frame": i, "timestamp": f"{i/30:.3f}",
                            "crop_x": 0, "crop_y": 0, "crop_w": 320, "crop_h": 180,
                            "home_score": 0, "away_score": 0, "clock_running": "false",
                            "clock_seconds": 0, "half": 1, "scoreboard_visible": "true"})

        output_path = str(tmp_path / "out.mp4")
        render_broadcast(video_path, log_path, output_path,
                         output_width=320, output_height=180)
        assert os.path.exists(output_path)

        # Output should have 3 frames (video length), not 10
        cap = cv2.VideoCapture(output_path)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert fc == 3


class TestJoystickDisconnect:
    def test_reconnect_method_exists(self):
        """Verify InteractiveViewer has _try_reconnect_joystick method."""
        from interactive import InteractiveViewer
        assert hasattr(InteractiveViewer, "_try_reconnect_joystick")

    def test_handle_joystick_with_no_joystick(self, tmp_path):
        """Verify handle_joystick is safe with no joystick."""
        from interactive import InteractiveViewer
        video_path = make_test_video(tmp_path)
        viewer = InteractiveViewer(video_path)
        viewer.joystick = None
        # Should not raise
        viewer.handle_joystick()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
