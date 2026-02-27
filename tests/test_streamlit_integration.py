"""
Phase 4 Streamlit Integration Test

Verifies:
1. All page modules import cleanly
2. Config loads and has required fields
3. Stage file detection works across all stages
4. Session summary parsing from interactive log
5. Render log summary parsing
6. Full data flow: stitch → interactive → render functions are compatible
"""

import csv
import os
import sys

import cv2
import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pages"))


class TestModuleImports:
    def test_app_imports(self):
        """Verify app.py imports without error."""
        from app import load_config, check_stage_files, render_sidebar
        assert callable(load_config)
        assert callable(check_stage_files)

    def test_stitch_page_imports(self):
        """Verify Stitch page imports without error."""
        from importlib import import_module
        stitch_page = import_module("1_Stitch")
        assert hasattr(stitch_page, "format_duration")
        assert hasattr(stitch_page, "get_stitch_preview_frame")

    def test_interactive_page_imports(self):
        """Verify Interactive page imports without error."""
        from importlib import import_module
        interactive_page = import_module("2_Interactive")
        assert hasattr(interactive_page, "get_session_summary")
        assert hasattr(interactive_page, "CONTROLS_CARD")

    def test_render_page_imports(self):
        """Verify Render page imports without error."""
        from importlib import import_module
        render_page = import_module("3_Render")
        assert hasattr(render_page, "read_log_summary")
        assert hasattr(render_page, "format_duration")


class TestConfigIntegration:
    def test_config_loads(self):
        """Verify config.yaml loads with all required sections."""
        from app import load_config
        config = load_config()
        assert "home_team" in config
        assert "away_team" in config
        assert "controller" in config
        assert "output_width" in config
        assert "scoreboard_font" in config

    def test_config_values_usable_by_interactive(self):
        """Verify config values are compatible with InteractiveViewer."""
        from app import load_config
        config = load_config()
        # These should work as InteractiveViewer config
        assert isinstance(config.get("home_team"), str)
        assert isinstance(config.get("home_color"), str)
        assert config["home_color"].startswith("#")
        assert isinstance(config.get("controller"), dict)


class TestStageFileDetection:
    def test_all_stages_detected(self, tmp_path):
        """Verify check_stage_files detects files via session state."""
        import streamlit as st
        from app import check_stage_files

        # Create temp files
        stitched = str(tmp_path / "stitched.mp4")
        log = str(tmp_path / "log.csv")
        broadcast = str(tmp_path / "broadcast.mp4")

        for path in [stitched, log, broadcast]:
            with open(path, "w") as f:
                f.write("test")

        st.session_state["stitched_path"] = stitched
        st.session_state["log_path"] = log
        st.session_state["broadcast_path"] = broadcast

        stages = check_stage_files("test-date")
        assert stages["stitched"] is True
        assert stages["log"] is True
        assert stages["broadcast"] is True

        # Clean up
        for key in ["stitched_path", "log_path", "broadcast_path"]:
            del st.session_state[key]


class TestDataFlowIntegration:
    def test_stitch_to_interactive_to_render(self, tmp_path):
        """End-to-end: stitch output → interactive session → render input.

        Tests that the data formats are compatible across all three stages.
        """
        from calibrate import calibrate
        from stitch import stitch_videos, get_video_info
        from interactive import InteractiveViewer
        from render import render_broadcast

        # Create test video pair
        width, height, overlap = 400, 300, 120
        full_width = 2 * width - overlap
        rng = np.random.RandomState(42)
        base = rng.randint(50, 200, (height, full_width, 3), dtype=np.uint8)
        for _ in range(15):
            cx, cy = rng.randint(0, full_width), rng.randint(0, height)
            cv2.circle(base, (cx, cy), 15, tuple(int(c) for c in rng.randint(0, 256, 3)), -1)
        base = cv2.GaussianBlur(base, (5, 5), 1.0)

        left_path = str(tmp_path / "left.mp4")
        right_path = str(tmp_path / "right.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wl = cv2.VideoWriter(left_path, fourcc, 30.0, (width, height))
        wr = cv2.VideoWriter(right_path, fourcc, 30.0, (width, height))
        for i in range(20):
            noise = rng.randint(-3, 4, base.shape, dtype=np.int16)
            frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            wl.write(frame[:, :width])
            wr.write(frame[:, (width - overlap):])
        wl.release()
        wr.release()

        # Stage 1: Stitch
        cal_data = calibrate(left_path, right_path, cal_date="flow-test",
                            output_dir=str(tmp_path / "cal"), frame_index=0,
                            overlap_fraction=0.4)
        cal_path = str(tmp_path / "cal" / "flow-test_cal.json")

        stitched_path = str(tmp_path / "stitched.mp4")
        stitch_videos(left_path, right_path, stitched_path,
                      cal_path=cal_path, frame_offset=0)
        assert os.path.exists(stitched_path)

        # Stage 2: Interactive (headless)
        viewer = InteractiveViewer(stitched_path, config={
            "home_team": "Eagles", "away_team": "Hawks"
        })
        viewer.run(headless=True)

        log_path = str(tmp_path / "session.csv")
        viewer.save_log(log_path)
        assert os.path.exists(log_path)

        # Verify log is readable by render page summary
        from importlib import import_module
        render_page = import_module("3_Render")
        summary = render_page.read_log_summary(log_path)
        assert summary is not None
        assert summary["total_frames"] == 20

        # Verify log is readable by interactive page summary
        interactive_page = import_module("2_Interactive")
        isummary = interactive_page.get_session_summary(log_path)
        assert isummary is not None
        assert isummary["total_frames"] == 20

        # Stage 3: Render
        broadcast_path = str(tmp_path / "broadcast.mp4")
        render_broadcast(stitched_path, log_path, broadcast_path,
                         home_team="Eagles", away_team="Hawks",
                         output_width=640, output_height=360)
        assert os.path.exists(broadcast_path)

        # Verify output
        cap = cv2.VideoCapture(broadcast_path)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert w == 640
        assert h == 360
        assert fc == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
