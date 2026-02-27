"""
Page 2 — Interactive PTZ Session

Match setup (team names, colors), controls reference card,
launch pygame as subprocess, session summary on return.
"""

import csv
import os
import subprocess
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import render_sidebar, load_config


def get_session_summary(log_path):
    """Parse a session log and return summary stats."""
    if not os.path.exists(log_path):
        return None

    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    total_frames = len(rows)
    last_row = rows[-1]

    # Count score changes (skip first row as baseline)
    score_changes = 0
    prev_home = rows[0].get("home_score", "0")
    prev_away = rows[0].get("away_score", "0")
    for row in rows[1:]:
        if row["home_score"] != prev_home or row["away_score"] != prev_away:
            score_changes += 1
            prev_home = row["home_score"]
            prev_away = row["away_score"]

    # Get final timestamp
    try:
        duration = float(last_row.get("timestamp", 0))
    except (ValueError, TypeError):
        duration = total_frames / 30.0

    return {
        "total_frames": total_frames,
        "duration": duration,
        "home_score": last_row.get("home_score", "0"),
        "away_score": last_row.get("away_score", "0"),
        "score_changes": score_changes,
        "half": last_row.get("half", "1"),
        "log_path": log_path,
    }


def format_duration(seconds):
    """Format seconds as M:SS or H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


CONTROLS_CARD = """
| **Camera (Switch Pro Controller)** | **Scoreboard (Keyboard)** |
|:---|:---|
| Left Stick &mdash; Pan / Tilt | Space &mdash; Start/Stop clock |
| ZR trigger &mdash; Zoom in | Q / A &mdash; Home +1 / -1 |
| ZL trigger &mdash; Zoom out | P / L &mdash; Away +1 / -1 |
| B button &mdash; Snap center | H &mdash; Switch half + reset clock |
| A button &mdash; Snap left goal | R &mdash; Reset clock |
| Y button &mdash; Snap right goal | T &mdash; Toggle scoreboard |
| X button &mdash; Wide view | Esc &mdash; Quit + save |
"""


def main():
    st.set_page_config(page_title="Interactive - Soccer Broadcast", layout="wide")
    render_sidebar()

    st.title("Interactive PTZ Session")

    config = load_config()

    # Check for stitched video
    stitched_path = st.session_state.get("stitched_path", "")
    if not stitched_path:
        default_stitched = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "match_stitched.mp4"
        )
        if os.path.exists(default_stitched):
            stitched_path = default_stitched

    stitched_path = st.text_input(
        "Stitched Video Path",
        value=stitched_path,
        placeholder="output/match_stitched.mp4"
    )

    if stitched_path and not os.path.exists(stitched_path):
        st.warning(f"Stitched video not found: {stitched_path}")
        st.info("Complete Stage 1 (Stitch) first, or enter the path to a stitched video.")

    st.markdown("---")

    # Match Setup
    st.subheader("Match Setup")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("Home Team", value=st.session_state.get(
            "home_team", config.get("home_team", "HOME")))
        home_color = st.color_picker("Home Color", value=st.session_state.get(
            "home_color", config.get("home_color", "#1E3A5F")))
    with col2:
        away_team = st.text_input("Away Team", value=st.session_state.get(
            "away_team", config.get("away_team", "AWAY")))
        away_color = st.color_picker("Away Color", value=st.session_state.get(
            "away_color", config.get("away_color", "#8B0000")))

    # Save to session state
    st.session_state["home_team"] = home_team
    st.session_state["away_team"] = away_team
    st.session_state["home_color"] = home_color
    st.session_state["away_color"] = away_color

    st.markdown("---")

    # Controls Reference Card
    st.subheader("Controls Reference")
    st.markdown(CONTROLS_CARD)

    st.markdown("---")

    # Log output path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_log = os.path.join(base_dir, "logs", "match.csv")
    log_output = st.text_input("Session Log Output Path", value=default_log)

    # Launch button
    if st.button("Launch Interactive Session", type="primary",
                 disabled=not (stitched_path and os.path.exists(stitched_path))):
        st.info("Session in progress... close the pygame window when done.")

        # Build the command
        script_path = os.path.join(base_dir, "interactive.py")
        cmd = [
            sys.executable, script_path,
            "--video", stitched_path,
            "--log-output", log_output,
        ]

        # Set environment variables for match config
        env = os.environ.copy()
        env["MATCH_HOME_TEAM"] = home_team
        env["MATCH_AWAY_TEAM"] = away_team
        env["MATCH_HOME_COLOR"] = home_color
        env["MATCH_AWAY_COLOR"] = away_color

        try:
            result = subprocess.run(cmd, env=env, timeout=7200)  # 2 hour timeout
            if result.returncode == 0:
                st.success("Session complete!")
                st.session_state["log_path"] = log_output
            else:
                st.warning(f"Session ended with code {result.returncode}")
        except subprocess.TimeoutExpired:
            st.warning("Session timed out after 2 hours.")
        except Exception as e:
            st.error(f"Failed to launch session: {e}")

    # Session Summary
    log_path = st.session_state.get("log_path", log_output)
    if log_path and os.path.exists(log_path):
        st.markdown("---")
        st.subheader("Session Summary")
        summary = get_session_summary(log_path)
        if summary:
            col1, col2, col3 = st.columns(3)
            col1.metric("Duration", format_duration(summary["duration"]))
            col2.metric("Final Score",
                        f"{summary['home_score']} - {summary['away_score']}")
            col3.metric("Score Changes", str(summary["score_changes"]))
            st.markdown(f"**Log file:** `{summary['log_path']}`")
            st.markdown(f"**Total frames:** {summary['total_frames']:,}")


main()
