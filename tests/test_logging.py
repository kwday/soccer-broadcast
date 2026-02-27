"""
Tests for CSV frame logging in interactive.py

Verifies:
1. CSV file is written with correct structure
2. Column count matches expected
3. Timestamps increment monotonically
4. Score changes appear at correct frames
5. All expected columns are present
"""

import csv
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interactive import InteractiveViewer


def make_test_video(tmp_path, width=800, height=400, num_frames=30, fps=30.0):
    path = str(tmp_path / "test_pano.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.RandomState(42)
    for i in range(num_frames):
        frame = rng.randint(50, 200, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


EXPECTED_COLUMNS = [
    "frame", "timestamp", "crop_x", "crop_y", "crop_w", "crop_h",
    "home_score", "away_score", "clock_running", "clock_seconds",
    "half", "scoreboard_visible"
]


class TestCSVLogging:
    def test_csv_file_is_written(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=10)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        log_path = str(tmp_path / "test_log.csv")
        result = viewer.save_log(log_path)
        assert result == log_path
        assert os.path.exists(log_path)

    def test_csv_has_correct_columns(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=5)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        log_path = str(tmp_path / "test_log.csv")
        viewer.save_log(log_path)

        with open(log_path) as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == EXPECTED_COLUMNS

    def test_csv_row_count_matches_frames(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=15)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        log_path = str(tmp_path / "test_log.csv")
        viewer.save_log(log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 15

    def test_timestamps_increment(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=20, fps=30.0)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        log_path = str(tmp_path / "test_log.csv")
        viewer.save_log(log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        timestamps = [float(r["timestamp"]) for r in rows]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1], \
                f"Timestamp not increasing at row {i}: {timestamps[i]} <= {timestamps[i-1]}"

    def test_frame_numbers_sequential(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=10)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        log_path = str(tmp_path / "test_log.csv")
        viewer.save_log(log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for i, row in enumerate(rows):
            assert int(row["frame"]) == i

    def test_score_changes_recorded(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=10)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()

        # Simulate: frames 0-2 at 0-0, frame 3 home scores, frame 6 away scores
        for i in range(10):
            ret, frame = viewer.cap.read()
            if not ret:
                break
            if i == 3:
                viewer.home_score = 1
            if i == 6:
                viewer.away_score = 1
            viewer.log_frame(i / viewer.fps)
            viewer.current_frame += 1

        viewer.cap.release()

        log_path = str(tmp_path / "test_log.csv")
        viewer.save_log(log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Score should change at frame 3 and 6
        assert rows[2]["home_score"] == "0"
        assert rows[3]["home_score"] == "1"
        assert rows[5]["away_score"] == "0"
        assert rows[6]["away_score"] == "1"

    def test_clock_running_recorded(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=10)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()

        for i in range(10):
            ret, frame = viewer.cap.read()
            if not ret:
                break
            if i == 3:
                viewer.clock_running = True
            if viewer.clock_running:
                viewer.clock_seconds += 1.0 / viewer.fps
            viewer.log_frame(i / viewer.fps)
            viewer.current_frame += 1

        viewer.cap.release()

        log_path = str(tmp_path / "test_log.csv")
        viewer.save_log(log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[2]["clock_running"] == "false"
        assert rows[3]["clock_running"] == "true"

    def test_half_change_recorded(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=10)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()

        for i in range(10):
            ret, frame = viewer.cap.read()
            if not ret:
                break
            if i == 5:
                viewer.half = 2
                viewer.clock_seconds = 0.0
            viewer.log_frame(i / viewer.fps)
            viewer.current_frame += 1

        viewer.cap.release()

        log_path = str(tmp_path / "test_log.csv")
        viewer.save_log(log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[4]["half"] == "1"
        assert rows[5]["half"] == "2"

    def test_crop_coords_in_valid_range(self, tmp_path):
        video_path = make_test_video(tmp_path, width=800, height=400, num_frames=5)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        for row in viewer.log_rows:
            assert int(row["crop_x"]) >= 0
            assert int(row["crop_y"]) >= 0
            assert int(row["crop_w"]) > 0
            assert int(row["crop_h"]) > 0

    def test_column_count_per_row(self, tmp_path):
        video_path = make_test_video(tmp_path, num_frames=5)
        viewer = InteractiveViewer(video_path)
        viewer.run(headless=True)

        log_path = str(tmp_path / "test_log.csv")
        viewer.save_log(log_path)

        with open(log_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_count = len(header)
            for i, row in enumerate(reader):
                assert len(row) == expected_count, \
                    f"Row {i} has {len(row)} columns, expected {expected_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
