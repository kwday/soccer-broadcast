"""
Tests for pages/3_Render.py

Verifies:
1. Log summary parsing is correct
2. Duration formatting works
3. Summary reads from log correctly
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pages"))

from importlib import import_module
render_page = import_module("3_Render")
read_log_summary = render_page.read_log_summary
format_duration = render_page.format_duration


def make_test_log(tmp_path, num_frames=30, home_score=2, away_score=1):
    """Create a test session log."""
    path = str(tmp_path / "session.csv")
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
                "crop_x": 100, "crop_y": 100,
                "crop_w": 960, "crop_h": 540,
                "home_score": home_score if i >= num_frames // 2 else 0,
                "away_score": away_score if i >= num_frames * 3 // 4 else 0,
                "clock_running": "true",
                "clock_seconds": i,
                "half": 2 if i >= num_frames // 2 else 1,
                "scoreboard_visible": "true",
            })
    return path


class TestReadLogSummary:
    def test_reads_correct_frame_count(self, tmp_path):
        log_path = make_test_log(tmp_path, num_frames=60)
        summary = read_log_summary(log_path)
        assert summary["total_frames"] == 60

    def test_reads_final_score(self, tmp_path):
        log_path = make_test_log(tmp_path, home_score=3, away_score=2)
        summary = read_log_summary(log_path)
        assert summary["home_score"] == "3"
        assert summary["away_score"] == "2"

    def test_reads_duration(self, tmp_path):
        log_path = make_test_log(tmp_path, num_frames=90)
        summary = read_log_summary(log_path)
        assert summary["duration"] > 0

    def test_reads_half(self, tmp_path):
        log_path = make_test_log(tmp_path, num_frames=30)
        summary = read_log_summary(log_path)
        assert summary["half"] in ("1", "2")

    def test_nonexistent_returns_none(self):
        summary = read_log_summary("/nonexistent/log.csv")
        assert summary is None

    def test_empty_log_returns_none(self, tmp_path):
        path = str(tmp_path / "empty.csv")
        with open(path, "w") as f:
            f.write("frame,timestamp\n")
        summary = read_log_summary(path)
        assert summary is None


class TestFormatDuration:
    def test_short(self):
        assert format_duration(10) == "0:10"

    def test_minutes(self):
        assert format_duration(125) == "2:05"

    def test_long(self):
        assert format_duration(5400) == "1:30:00"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
