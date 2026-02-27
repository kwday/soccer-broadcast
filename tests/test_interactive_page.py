"""
Tests for pages/2_Interactive.py

Verifies:
1. Session summary parsing works correctly
2. Duration formatting is correct
3. Controls reference card content
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pages"))

from importlib import import_module
interactive_page = import_module("2_Interactive")
get_session_summary = interactive_page.get_session_summary
format_duration = interactive_page.format_duration
CONTROLS_CARD = interactive_page.CONTROLS_CARD


def make_test_log(tmp_path, num_frames=30, score_changes=None):
    """Create a test session log CSV."""
    path = str(tmp_path / "session.csv")
    fieldnames = [
        "frame", "timestamp", "crop_x", "crop_y", "crop_w", "crop_h",
        "home_score", "away_score", "clock_running", "clock_seconds",
        "half", "scoreboard_visible"
    ]

    if score_changes is None:
        score_changes = {10: (1, 0), 20: (1, 1)}

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        home, away = 0, 0
        for i in range(num_frames):
            if i in score_changes:
                home, away = score_changes[i]
            writer.writerow({
                "frame": i,
                "timestamp": f"{i/30:.3f}",
                "crop_x": 100,
                "crop_y": 100,
                "crop_w": 960,
                "crop_h": 540,
                "home_score": home,
                "away_score": away,
                "clock_running": "true",
                "clock_seconds": i,
                "half": 1 if i < 15 else 2,
                "scoreboard_visible": "true",
            })

    return path


class TestGetSessionSummary:
    def test_basic_summary(self, tmp_path):
        """Verify basic session summary fields."""
        log_path = make_test_log(tmp_path, num_frames=30)
        summary = get_session_summary(log_path)

        assert summary is not None
        assert summary["total_frames"] == 30
        assert summary["home_score"] == "1"
        assert summary["away_score"] == "1"
        assert summary["log_path"] == log_path

    def test_score_changes_counted(self, tmp_path):
        """Verify score change counting."""
        log_path = make_test_log(tmp_path, num_frames=30,
                                score_changes={5: (1, 0), 10: (2, 0), 20: (2, 1)})
        summary = get_session_summary(log_path)
        assert summary["score_changes"] == 3

    def test_no_score_changes(self, tmp_path):
        """Verify zero score changes when score never changes."""
        log_path = make_test_log(tmp_path, num_frames=10, score_changes={})
        summary = get_session_summary(log_path)
        assert summary["score_changes"] == 0

    def test_duration_from_timestamp(self, tmp_path):
        """Verify duration is read from the last timestamp."""
        log_path = make_test_log(tmp_path, num_frames=60)
        summary = get_session_summary(log_path)
        assert summary["duration"] > 0

    def test_nonexistent_log_returns_none(self):
        """Verify graceful handling of missing log."""
        summary = get_session_summary("/nonexistent/log.csv")
        assert summary is None


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(30) == "0:30"

    def test_minutes(self):
        assert format_duration(90) == "1:30"

    def test_hours(self):
        assert format_duration(3661) == "1:01:01"


class TestControlsCard:
    def test_contains_key_controls(self):
        """Verify controls card mentions all key controls."""
        assert "Pan" in CONTROLS_CARD
        assert "Zoom" in CONTROLS_CARD
        assert "Space" in CONTROLS_CARD
        assert "Esc" in CONTROLS_CARD
        assert "Snap center" in CONTROLS_CARD

    def test_has_both_sections(self):
        """Verify both camera and scoreboard sections present."""
        assert "Camera" in CONTROLS_CARD
        assert "Scoreboard" in CONTROLS_CARD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
