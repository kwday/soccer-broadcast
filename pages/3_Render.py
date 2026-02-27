"""
Page 3 — Render

Pre-render summary (score, duration, settings), time estimate,
progress bar, and output file path.
"""

import csv
import os
import subprocess
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import render_sidebar, load_config
from render import render_broadcast, mux_audio


def open_file_dialog(title="Select File", filetypes=None):
    """Open a native file dialog and return the selected path."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    if filetypes is None:
        filetypes = [
            ("Video files", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV"),
            ("All files", "*.*"),
        ]
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path if path else None


def read_log_summary(log_path):
    """Read log and return summary info for pre-render display."""
    if not os.path.exists(log_path):
        return None

    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    last_row = rows[-1]
    try:
        duration = float(last_row.get("timestamp", 0))
    except (ValueError, TypeError):
        duration = len(rows) / 30.0

    return {
        "total_frames": len(rows),
        "duration": duration,
        "home_score": last_row.get("home_score", "0"),
        "away_score": last_row.get("away_score", "0"),
        "half": last_row.get("half", "1"),
    }


def format_duration(seconds):
    """Format seconds as M:SS or H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def main():
    st.set_page_config(page_title="Render - Soccer Broadcast", layout="wide")
    render_sidebar()

    st.title("Render Final Broadcast")

    config = load_config()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Input paths with browse buttons
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Stitched Video**")
        if st.button("Browse...", key="browse_stitched"):
            path = open_file_dialog("Select Stitched Video")
            if path:
                st.session_state["stitched_path"] = path
        stitched_path = st.session_state.get("stitched_path", "")
        if stitched_path:
            st.markdown(f"`{stitched_path}`")

    with col2:
        st.markdown("**Session Log**")
        if st.button("Browse...", key="browse_log"):
            path = open_file_dialog("Select Session Log", filetypes=[
                ("CSV files", "*.csv *.CSV"),
                ("All files", "*.*"),
            ])
            if path:
                st.session_state["log_path"] = path
        log_path = st.session_state.get("log_path", "")
        if log_path:
            st.markdown(f"`{log_path}`")

    # Validate inputs
    stitched_ok = stitched_path and os.path.exists(stitched_path)
    log_ok = log_path and os.path.exists(log_path)

    if stitched_path and not stitched_ok:
        st.warning(f"Stitched video not found: {stitched_path}")
    if log_path and not log_ok:
        st.warning(f"Session log not found: {log_path}")

    # Pre-render summary
    if log_ok:
        st.markdown("---")
        st.subheader("Pre-Render Summary")

        summary = read_log_summary(log_path)
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Duration", format_duration(summary["duration"]))
            col2.metric("Final Score",
                        f"{summary['home_score']} - {summary['away_score']}")
            col3.metric("Total Frames", f"{summary['total_frames']:,}")
            col4.metric("Half", summary["half"])

            # Scoreboard settings
            home_team = st.session_state.get("home_team", config.get("home_team", "HOME"))
            away_team = st.session_state.get("away_team", config.get("away_team", "AWAY"))
            home_color = st.session_state.get("home_color", config.get("home_color", "#1E3A5F"))
            away_color = st.session_state.get("away_color", config.get("away_color", "#8B0000"))

            st.markdown(f"**Teams:** {home_team} vs {away_team}")
            st.markdown(f"**Colors:** {home_color} / {away_color}")

            # Time estimate
            est_minutes = summary["duration"] * 0.5 / 60
            st.markdown(f"**Estimated render time:** ~{max(1, est_minutes):.0f} minutes")

    st.markdown("---")

    # Output settings
    st.subheader("Output Settings")
    col1, col2 = st.columns(2)
    with col1:
        output_width = st.number_input("Width", value=config.get("output_width", 1920),
                                       min_value=640, max_value=3840, step=160)
        output_fps = st.number_input("FPS", value=float(config.get("output_fps", 30)),
                                     min_value=24.0, max_value=60.0, step=1.0)
    with col2:
        output_height = st.number_input("Height", value=config.get("output_height", 1080),
                                        min_value=360, max_value=2160, step=90)

    if "render_output_folder" not in st.session_state:
        st.session_state["render_output_folder"] = os.path.join(base_dir, "output")

    st.markdown("**Output Folder**")
    if st.button("Browse...", key="browse_render_output"):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        folder = filedialog.askdirectory(title="Select Output Folder")
        root.destroy()
        if folder:
            st.session_state["render_output_folder"] = folder
    st.markdown(f"`{st.session_state['render_output_folder']}`")
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = os.path.join(st.session_state["render_output_folder"], f"match_broadcast_{timestamp}.mp4")

    # Audio source — automatically uses left camera file from Stage 1
    audio_source = st.session_state.get("left_video_path", "")
    if audio_source and os.path.exists(audio_source):
        st.markdown(f"**Audio source:** `{os.path.basename(audio_source)}` (left camera)")
    else:
        st.info("Audio: select left camera video in Stage 1 (Stitch) to include audio in final output.")

    st.markdown("---")

    # Render button
    can_render = stitched_ok and log_ok
    if st.button("Render Final Broadcast", type="primary", disabled=not can_render):
        home_team = st.session_state.get("home_team", config.get("home_team", "HOME"))
        away_team = st.session_state.get("away_team", config.get("away_team", "AWAY"))
        home_color = st.session_state.get("home_color", config.get("home_color", "#1E3A5F"))
        away_color = st.session_state.get("away_color", config.get("away_color", "#8B0000"))

        progress_bar = st.progress(0, text="Starting render...")
        status_text = st.empty()

        def on_progress(current, total):
            pct = current / max(total, 1)
            progress_bar.progress(pct, text=f"Rendering frame {current:,} / {total:,}")

        try:
            result = render_broadcast(
                stitched_path, log_path, output_path,
                home_team=home_team, away_team=away_team,
                home_color=home_color, away_color=away_color,
                output_width=int(output_width), output_height=int(output_height),
                output_fps=float(output_fps),
                progress_callback=on_progress
            )

            progress_bar.progress(1.0, text="Render complete!")

            # Mux audio if source provided
            if audio_source and os.path.exists(audio_source):
                status_text.text("Muxing audio...")
                result = mux_audio(result, audio_source, result)
                status_text.empty()

            st.success(f"Broadcast rendered: `{result}`")
            st.session_state["broadcast_path"] = result

            # Open folder button
            output_dir = os.path.dirname(os.path.abspath(result))
            st.markdown(f"**Output folder:** `{output_dir}`")

        except Exception as e:
            st.error(f"Render failed: {e}")

    # Show previous render result
    if st.session_state.get("broadcast_path") and os.path.exists(
        st.session_state["broadcast_path"]
    ):
        st.markdown("---")
        st.success(f"Previous render: `{st.session_state['broadcast_path']}`")


main()
