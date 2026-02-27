"""
app.py — Streamlit entry point for the Soccer Broadcast System.

Provides the sidebar with at-a-glance match status and page routing.

Usage:
    streamlit run app.py
"""

import os
from datetime import date

import streamlit as st
import yaml


def load_config():
    """Load config.yaml, returning defaults if not found."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        "home_team": "HOME",
        "away_team": "AWAY",
        "home_color": "#1E3A5F",
        "away_color": "#8B0000",
        "output_width": 1920,
        "output_height": 1080,
        "output_fps": 30,
    }


def get_match_date():
    """Get the current match date from session state or default to today."""
    if "match_date" not in st.session_state:
        st.session_state.match_date = date.today().isoformat()
    return st.session_state.match_date


def check_stage_files(match_date):
    """Check which output files exist for the given match date."""
    base_dir = os.path.dirname(__file__)

    # Check for stitched video
    stitched = any(
        os.path.exists(os.path.join(base_dir, "output", f))
        for f in [
            f"{match_date}_stitched.mp4",
            "stitched.mp4",
        ]
    )
    # Also check session state for dynamically set paths
    if st.session_state.get("stitched_path") and os.path.exists(
        st.session_state["stitched_path"]
    ):
        stitched = True

    # Check for PTZ session log
    log = any(
        os.path.exists(os.path.join(base_dir, "logs", f))
        for f in [
            f"{match_date}_match.csv",
            "match.csv",
        ]
    )
    if st.session_state.get("log_path") and os.path.exists(
        st.session_state["log_path"]
    ):
        log = True

    # Check for final broadcast
    broadcast = any(
        os.path.exists(os.path.join(base_dir, "output", f))
        for f in [
            f"{match_date}_broadcast_1080p.mp4",
            "match_broadcast_1080p.mp4",
        ]
    )
    if st.session_state.get("broadcast_path") and os.path.exists(
        st.session_state["broadcast_path"]
    ):
        broadcast = True

    return {
        "stitched": stitched,
        "log": log,
        "broadcast": broadcast,
    }


def render_sidebar():
    """Render the persistent sidebar with match status."""
    config = load_config()
    match_date = get_match_date()
    stages = check_stage_files(match_date)

    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Match:** {match_date}")

        home = st.session_state.get("home_team", config.get("home_team", "HOME"))
        away = st.session_state.get("away_team", config.get("away_team", "AWAY"))
        st.markdown(f"**{home}** vs **{away}**")

        st.markdown("---")

        def status_icon(done):
            return "\u2713" if done else "\u2717"

        st.markdown(
            f"{status_icon(stages['stitched'])} Stitched video"
        )
        st.markdown(
            f"{status_icon(stages['log'])} PTZ session log"
        )
        st.markdown(
            f"{status_icon(stages['broadcast'])} Final broadcast"
        )

        st.markdown("---")


def main():
    st.set_page_config(
        page_title="Soccer Broadcast System",
        page_icon="",
        layout="wide",
    )

    render_sidebar()

    st.title("Soccer Broadcast System")
    st.markdown(
        "Welcome to the DIY Soccer Broadcast pipeline. "
        "Use the pages in the sidebar to work through each stage:"
    )
    st.markdown(
        """
        1. **Stitch** — Calibrate and stitch left + right camera feeds into a panorama
        2. **Interactive** — Run the PTZ session with joystick control and scoreboard
        3. **Render** — Encode the final broadcast output with audio
        """
    )

    config = load_config()
    match_date = get_match_date()
    stages = check_stage_files(match_date)

    # Summary cards
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "Complete" if stages["stitched"] else "Pending"
        st.metric("Stage 1: Stitch", status)
    with col2:
        status = "Complete" if stages["log"] else "Pending"
        st.metric("Stage 2: Interactive", status)
    with col3:
        status = "Complete" if stages["broadcast"] else "Pending"
        st.metric("Stage 3: Render", status)


if __name__ == "__main__":
    main()
