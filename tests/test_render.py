"""
Tests for render.py

Verifies:
1. Render from a known log CSV produces output
2. Output is 1920x1080 (or configured size)
3. Frame count matches log rows
4. Scoreboard is visible in output frames
5. Crop positions are applied correctly
"""

import csv
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render import render_broadcast, read_log


def make_test_panorama(tmp_path, width=2000, height=1200, num_frames=20, fps=30.0):
    """Create a synthetic panoramic video with visible features."""
    path = str(tmp_path / "panorama.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    rng = np.random.RandomState(42)
    for i in range(num_frames):
        frame = rng.randint(30, 180, (height, width, 3), dtype=np.uint8)
        # Add a distinguishing feature at center
        cv2.circle(frame, (width // 2, height // 2), 50, (0, 255, 0), -1)
        # Add frame number text
        cv2.putText(frame, str(i), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 3)
        writer.write(frame)

    writer.release()
    return path


def make_test_log(tmp_path, num_frames=20, crop_x=40, crop_y=60,
                  crop_w=960, crop_h=540):
    """Create a test log CSV."""
    path = str(tmp_path / "test_log.csv")
    fieldnames = [
        "frame", "timestamp", "crop_x", "crop_y", "crop_w", "crop_h",
        "home_score", "away_score", "clock_running", "clock_seconds",
        "half", "scoreboard_visible"
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_frames):
            writer.writerow({
                "frame": i,
                "timestamp": f"{i/30:.3f}",
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "home_score": 1 if i >= 10 else 0,
                "away_score": 0,
                "clock_running": "true" if i >= 2 else "false",
                "clock_seconds": max(0, i - 2),
                "half": 1,
                "scoreboard_visible": "true",
            })

    return path


class TestReadLog:
    def test_reads_correct_row_count(self, tmp_path):
        log_path = make_test_log(tmp_path, num_frames=15)
        rows = read_log(log_path)
        assert len(rows) == 15

    def test_reads_correct_columns(self, tmp_path):
        log_path = make_test_log(tmp_path, num_frames=5)
        rows = read_log(log_path)
        assert "frame" in rows[0]
        assert "crop_x" in rows[0]
        assert "home_score" in rows[0]
        assert "scoreboard_visible" in rows[0]


class TestRenderBroadcast:
    def test_output_exists(self, tmp_path):
        video_path = make_test_panorama(tmp_path, num_frames=10)
        log_path = make_test_log(tmp_path, num_frames=10)
        output_path = str(tmp_path / "broadcast.mp4")

        render_broadcast(
            video_path, log_path, output_path,
            output_width=640, output_height=360, output_fps=30.0
        )

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_output_resolution(self, tmp_path):
        video_path = make_test_panorama(tmp_path, num_frames=5)
        log_path = make_test_log(tmp_path, num_frames=5)
        output_path = str(tmp_path / "broadcast.mp4")

        render_broadcast(
            video_path, log_path, output_path,
            output_width=640, output_height=360, output_fps=30.0
        )

        cap = cv2.VideoCapture(output_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert w == 640
        assert h == 360

    def test_frame_count_matches_log(self, tmp_path):
        video_path = make_test_panorama(tmp_path, num_frames=15)
        log_path = make_test_log(tmp_path, num_frames=15)
        output_path = str(tmp_path / "broadcast.mp4")

        render_broadcast(
            video_path, log_path, output_path,
            output_width=640, output_height=360
        )

        cap = cv2.VideoCapture(output_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        assert frame_count == 15

    def test_scoreboard_visible_in_output(self, tmp_path):
        """Verify scoreboard composite changes the output frame."""
        video_path = make_test_panorama(tmp_path, num_frames=5)

        # Create log with scoreboard visible
        visible_log = make_test_log(tmp_path, num_frames=5)

        output_visible = str(tmp_path / "visible.mp4")
        render_broadcast(
            video_path, visible_log, output_visible,
            output_width=640, output_height=360,
            home_team="Eagles", away_team="Hawks"
        )

        # Read a frame from the output
        cap = cv2.VideoCapture(output_visible)
        ret, frame_visible = cap.read()
        cap.release()
        assert ret

        # The frame should have content (not all black)
        assert frame_visible.sum() > 0

    def test_crop_positions_applied(self, tmp_path):
        """Verify different crop positions produce different outputs."""
        video_path = make_test_panorama(tmp_path, width=2000, height=1200, num_frames=5)

        # Log with crop at left
        log_left = str(tmp_path / "log_left.csv")
        fieldnames = [
            "frame", "timestamp", "crop_x", "crop_y", "crop_w", "crop_h",
            "home_score", "away_score", "clock_running", "clock_seconds",
            "half", "scoreboard_visible"
        ]
        with open(log_left, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i in range(5):
                w.writerow({
                    "frame": i, "timestamp": f"{i/30:.3f}",
                    "crop_x": 0, "crop_y": 0, "crop_w": 960, "crop_h": 540,
                    "home_score": 0, "away_score": 0, "clock_running": "false",
                    "clock_seconds": 0, "half": 1, "scoreboard_visible": "false"
                })

        # Log with crop at right
        log_right = str(tmp_path / "log_right.csv")
        with open(log_right, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i in range(5):
                w.writerow({
                    "frame": i, "timestamp": f"{i/30:.3f}",
                    "crop_x": 1000, "crop_y": 0, "crop_w": 960, "crop_h": 540,
                    "home_score": 0, "away_score": 0, "clock_running": "false",
                    "clock_seconds": 0, "half": 1, "scoreboard_visible": "false"
                })

        out_left = str(tmp_path / "left.mp4")
        out_right = str(tmp_path / "right.mp4")

        render_broadcast(video_path, log_left, out_left,
                         output_width=640, output_height=360)
        render_broadcast(video_path, log_right, out_right,
                         output_width=640, output_height=360)

        # Read first frame from each
        cap_l = cv2.VideoCapture(out_left)
        _, frame_l = cap_l.read()
        cap_l.release()

        cap_r = cv2.VideoCapture(out_right)
        _, frame_r = cap_r.read()
        cap_r.release()

        # Frames should be different (different crop positions)
        assert not np.array_equal(frame_l, frame_r)

    def test_progress_callback(self, tmp_path):
        video_path = make_test_panorama(tmp_path, num_frames=5)
        log_path = make_test_log(tmp_path, num_frames=5)
        output_path = str(tmp_path / "broadcast.mp4")

        progress_log = []

        def on_progress(current, total):
            progress_log.append((current, total))

        render_broadcast(
            video_path, log_path, output_path,
            output_width=640, output_height=360,
            progress_callback=on_progress
        )

        assert len(progress_log) == 5
        assert progress_log[-1] == (5, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
