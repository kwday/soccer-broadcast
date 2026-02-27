"""
Phase 2 Integration Test

Runs a full interactive session (headless) on a sample video and verifies:
1. All controls work together (crop movement, zoom, scoreboard)
2. Log file is complete and consistent
3. Scoreboard state changes are recorded
4. Crop coordinates stay within bounds
"""

import csv
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interactive import InteractiveViewer


def make_panorama_video(tmp_path, width=1600, height=600, num_frames=60, fps=30.0):
    """Create a synthetic panorama-like video."""
    path = str(tmp_path / "test_panorama.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    rng = np.random.RandomState(42)
    base = rng.randint(30, 180, (height, width, 3), dtype=np.uint8)

    # Add field-like features
    cv2.line(base, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
    cv2.circle(base, (width // 2, height // 2), 80, (255, 255, 255), 2)

    for i in range(num_frames):
        noise = rng.randint(-2, 3, base.shape, dtype=np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Moving ball
        bx = int(width * (0.3 + 0.4 * np.sin(2 * np.pi * i / num_frames)))
        by = height // 2
        cv2.circle(frame, (bx, by), 10, (0, 255, 255), -1)
        writer.write(frame)

    writer.release()
    return path


class TestPhase2Integration:
    def test_full_session_headless(self, tmp_path):
        """Run a full headless session and verify all state."""
        video_path = make_panorama_video(tmp_path, num_frames=60)
        viewer = InteractiveViewer(video_path, config={
            "home_team": "Eagles",
            "away_team": "Hawks",
            "home_color": "#1E5E3A",
            "away_color": "#5E1E1E",
        })

        viewer.run(headless=True)

        assert viewer.current_frame == 60
        assert len(viewer.log_rows) == 60

    def test_session_with_score_changes(self, tmp_path):
        """Simulate score changes during a session."""
        video_path = make_panorama_video(tmp_path, num_frames=30)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()
        viewer.running = True

        for i in range(30):
            ret, frame = viewer.cap.read()
            if not ret:
                break
            # Simulate events
            if i == 5:
                viewer.clock_running = True
            if i == 10:
                viewer.home_score = 1
            if i == 15:
                viewer.half = 2
                viewer.clock_seconds = 0.0
                viewer.clock_running = True
            if i == 20:
                viewer.away_score = 1
            if i == 25:
                viewer.home_score = 2

            if viewer.clock_running:
                viewer.clock_seconds += 1.0 / viewer.fps
            viewer.log_frame(i / viewer.fps)
            viewer.current_frame += 1

        viewer.cap.release()

        log_path = str(tmp_path / "session.csv")
        viewer.save_log(log_path)

        # Verify log
        with open(log_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 30

        # Score at various points
        assert rows[9]["home_score"] == "0"
        assert rows[10]["home_score"] == "1"
        assert rows[19]["away_score"] == "0"
        assert rows[20]["away_score"] == "1"
        assert rows[25]["home_score"] == "2"

        # Half change
        assert rows[14]["half"] == "1"
        assert rows[15]["half"] == "2"

    def test_crop_movement_recorded(self, tmp_path):
        """Verify crop position changes are logged."""
        # Use a wide panorama so crop can actually move
        video_path = make_panorama_video(tmp_path, width=4000, height=1500, num_frames=20)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()
        viewer.running = True

        for i in range(20):
            ret, frame = viewer.cap.read()
            if not ret:
                break
            # Move crop to the right each frame
            if i > 5:
                viewer.crop.move(20, 0)
            viewer.log_frame(i / viewer.fps)
            viewer.current_frame += 1

        viewer.cap.release()

        # Verify crop_x increases after frame 5
        x_values = [int(r["crop_x"]) for r in viewer.log_rows]
        assert x_values[10] > x_values[5], "Crop should have moved right"

    def test_zoom_changes_recorded(self, tmp_path):
        """Verify zoom (crop_w/crop_h) changes are logged."""
        video_path = make_panorama_video(tmp_path, num_frames=20)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()
        viewer.running = True

        for i in range(20):
            ret, frame = viewer.cap.read()
            if not ret:
                break
            if i == 10:
                viewer.crop.adjust_zoom(0.1)  # Zoom in
            viewer.log_frame(i / viewer.fps)
            viewer.current_frame += 1

        viewer.cap.release()

        # After zoom in, crop_w should decrease
        w_before = int(viewer.log_rows[9]["crop_w"])
        w_after = int(viewer.log_rows[10]["crop_w"])
        assert w_after < w_before, f"Expected crop_w to decrease: {w_before} -> {w_after}"

    def test_log_all_bounds_valid(self, tmp_path):
        """Verify all crop coords stay within panorama bounds."""
        pano_w, pano_h = 4000, 1500
        video_path = make_panorama_video(tmp_path, width=pano_w, height=pano_h, num_frames=30)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()
        viewer.running = True

        for i in range(30):
            ret, frame = viewer.cap.read()
            if not ret:
                break
            # Aggressive movement
            viewer.crop.move(50 if i % 2 == 0 else -50, 20 if i % 3 == 0 else -20)
            viewer.log_frame(i / viewer.fps)
            viewer.current_frame += 1

        viewer.cap.release()

        for row in viewer.log_rows:
            x = int(row["crop_x"])
            y = int(row["crop_y"])
            w = int(row["crop_w"])
            h = int(row["crop_h"])
            assert x >= 0
            assert y >= 0
            assert x + w <= pano_w
            assert y + h <= pano_h

    def test_scoreboard_renderer_works_in_session(self, tmp_path):
        """Verify scoreboard rendering doesn't crash during session."""
        video_path = make_panorama_video(tmp_path, num_frames=10)
        viewer = InteractiveViewer(video_path)
        viewer.open_video()

        # Verify scoreboard renderer was initialized
        assert viewer.scoreboard_renderer is not None

        # Test rendering
        state = viewer.get_scoreboard_state()
        from scoreboard import ScoreboardRenderer
        assert isinstance(viewer.scoreboard_renderer, ScoreboardRenderer)

        viewer.cap.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
