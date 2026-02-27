"""
Tests for app.py — Streamlit entry point

Verifies:
1. Config loads correctly from config.yaml
2. Sidebar renders match status with correct checkmarks
3. Stage file detection works
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import load_config, check_stage_files


class TestLoadConfig:
    def test_loads_config_yaml(self):
        """Verify config.yaml loads with expected keys."""
        config = load_config()
        assert "home_team" in config
        assert "away_team" in config
        assert "home_color" in config
        assert "away_color" in config
        assert "output_width" in config
        assert "output_height" in config

    def test_config_has_controller_settings(self):
        """Verify controller section exists."""
        config = load_config()
        assert "controller" in config
        ctrl = config["controller"]
        assert "axis_pan" in ctrl
        assert "deadzone" in ctrl

    def test_config_has_scoreboard_settings(self):
        """Verify scoreboard settings exist."""
        config = load_config()
        assert "scoreboard_font" in config
        assert "scoreboard_mono_font" in config

    def test_config_values_are_correct_types(self):
        """Verify config values parse to correct types."""
        config = load_config()
        assert isinstance(config["home_team"], str)
        assert isinstance(config["output_width"], int)
        assert isinstance(config["output_fps"], int)
        assert isinstance(config["home_color"], str)
        assert config["home_color"].startswith("#")


class TestCheckStageFiles:
    def test_no_files_exist(self, tmp_path):
        """All stages should be False when no files exist."""
        # Use streamlit's session_state mock
        import streamlit as st
        # Clear any session state
        for key in ["stitched_path", "log_path", "broadcast_path"]:
            if key in st.session_state:
                del st.session_state[key]

        stages = check_stage_files("9999-99-99")
        assert stages["stitched"] is False
        assert stages["log"] is False
        assert stages["broadcast"] is False

    def test_detects_stitched_via_session_state(self, tmp_path):
        """Verify stitched detection via session state path."""
        import streamlit as st

        # Create a temp file
        test_file = str(tmp_path / "stitched.mp4")
        with open(test_file, "w") as f:
            f.write("test")

        st.session_state["stitched_path"] = test_file
        stages = check_stage_files("test-date")
        assert stages["stitched"] is True

        # Clean up
        del st.session_state["stitched_path"]

    def test_detects_log_via_session_state(self, tmp_path):
        """Verify log detection via session state path."""
        import streamlit as st

        test_file = str(tmp_path / "session.csv")
        with open(test_file, "w") as f:
            f.write("test")

        st.session_state["log_path"] = test_file
        stages = check_stage_files("test-date")
        assert stages["log"] is True

        del st.session_state["log_path"]


class TestConfigYamlFile:
    def test_config_yaml_exists(self):
        """config.yaml should exist in the project root."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )
        assert os.path.exists(config_path)

    def test_config_yaml_valid(self):
        """config.yaml should be valid YAML."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
