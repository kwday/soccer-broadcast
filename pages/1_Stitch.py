"""
Page 1 — Calibrate + Stitch

Two-phase flow:
  Phase A: Calibrate + Preview — multi-candidate calibration, preview frame,
           quality metrics, "Try Another" to cycle candidates.
  Phase B: Approve + Stitch All Frames — full stitch with precomputed remap.
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stitch import get_video_info, stitch_videos, stitch_frame, detect_timecode_offset
from calibrate import calibrate_multi, extract_frame
from app import render_sidebar, load_config


def format_duration(seconds):
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def open_file_dialog(title="Select Video File"):
    """Open a native file dialog and return the selected path."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Video files", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV"),
            ("All files", "*.*"),
        ]
    )
    root.destroy()
    return path if path else None


def display_video_metadata(path, label):
    """Display video file metadata in a card."""
    try:
        info = get_video_info(path)
        duration_sec = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
        st.markdown(f"**{label}:** `{os.path.basename(path)}`")
        col1, col2, col3 = st.columns(3)
        col1.metric("Resolution", f"{info['width']}x{info['height']}")
        col2.metric("Duration", format_duration(duration_sec))
        col3.metric("Frame Rate", f"{info['fps']:.0f} fps")
        return info
    except Exception as e:
        st.error(f"Error reading {label}: {e}")
        return None


def detect_sync_method(left_path, right_path):
    """Detect sync method and display result."""
    tc_offset = detect_timecode_offset(left_path, right_path)
    if tc_offset is not None:
        st.success(f"Timecode sync detected (offset: {tc_offset:+.4f}s)")
        return "timecode", tc_offset
    else:
        st.info("No timecode — will use audio sync")
        return "audio", None


def get_stitch_preview_frame(video_path, frame_index=0):
    """Extract a single frame from a video for preview."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def generate_preview(left_path, right_path, cal_data):
    """Stitch a single preview frame using the given calibration data."""
    frame_idx = cal_data.get("frame_index", 0)
    img_l = extract_frame(left_path, frame_idx)
    img_r = extract_frame(right_path, frame_idx)

    H = np.array(cal_data["homography"], dtype=np.float64)
    T = np.array([[1, 0, -cal_data["offset_x"]],
                  [0, 1, -cal_data["offset_y"]],
                  [0, 0, 1]], dtype=np.float64)
    H_adj = T @ H

    stitched = stitch_frame(
        img_l, img_r, H_adj,
        cal_data["canvas_width"], cal_data["canvas_height"],
        cal_data["offset_x"], cal_data["offset_y"],
        cal_data["blend_x_start"], cal_data["blend_x_end"]
    )
    return cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)


def main():
    st.set_page_config(page_title="Stitch - Soccer Broadcast", layout="wide")
    render_sidebar()

    st.title("Calibrate + Stitch")
    st.markdown("Select left and right camera files, preview the calibration, then stitch.")

    # --- File selection ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Left Camera**")
        if st.button("Browse...", key="browse_left"):
            path = open_file_dialog("Select Left Camera Video")
            if path:
                st.session_state["left_video_path"] = path
        left_path = st.session_state.get("left_video_path", "")
        if left_path:
            st.markdown(f"`{left_path}`")

    with col2:
        st.markdown("**Right Camera**")
        if st.button("Browse...", key="browse_right"):
            path = open_file_dialog("Select Right Camera Video")
            if path:
                st.session_state["right_video_path"] = path
        right_path = st.session_state.get("right_video_path", "")
        if right_path:
            st.markdown(f"`{right_path}`")

    # --- Metadata display ---
    left_info = None
    right_info = None

    if left_path and os.path.exists(left_path):
        left_info = display_video_metadata(left_path, "Left Camera")
    elif left_path:
        st.warning(f"Left camera file not found: {left_path}")

    if right_path and os.path.exists(right_path):
        right_info = display_video_metadata(right_path, "Right Camera")
    elif right_path:
        st.warning(f"Right camera file not found: {right_path}")

    if not (left_info and right_info):
        return

    # --- Sync detection ---
    st.markdown("---")
    st.subheader("Sync Detection")
    sync_method, tc_offset = detect_sync_method(left_path, right_path)

    duration_sec = left_info["frame_count"] / left_info["fps"] if left_info["fps"] > 0 else 0
    est_minutes = duration_sec * 0.2 / 60
    st.markdown(f"**Estimated stitch time:** ~{est_minutes:.0f} minutes")

    st.markdown("---")

    # --- Output folder ---
    config = load_config()
    default_folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output"
    )
    if "stitch_output_folder" not in st.session_state:
        st.session_state["stitch_output_folder"] = default_folder

    st.markdown("**Output Folder**")
    if st.button("Browse...", key="browse_output"):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        folder = filedialog.askdirectory(title="Select Output Folder")
        root.destroy()
        if folder:
            st.session_state["stitch_output_folder"] = folder
    st.markdown(f"`{st.session_state['stitch_output_folder']}`")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = os.path.join(
        st.session_state["stitch_output_folder"],
        f"match_stitched_{timestamp}.mp4"
    )

    # =====================================================================
    # Phase A — Calibrate + Preview
    # =====================================================================
    has_candidates = "calibration_candidates" in st.session_state

    if not has_candidates:
        if st.button("Calibrate + Preview", type="primary"):
            with st.spinner("Running multi-candidate calibration (trying 4 frames)..."):
                from datetime import datetime as dt
                cal_date = dt.now().strftime("%Y-%m-%d")
                try:
                    candidates = calibrate_multi(
                        left_path, right_path, cal_date=cal_date
                    )
                    st.session_state["calibration_candidates"] = candidates
                    st.session_state["current_candidate_idx"] = 0
                    st.session_state["preview_frame"] = generate_preview(
                        left_path, right_path, candidates[0]
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Calibration failed: {e}")

        # Show previous stitch result if available
        if st.session_state.get("stitched_path") and \
           os.path.exists(st.session_state["stitched_path"]):
            st.markdown("---")
            st.success(f"Previous stitch: `{st.session_state['stitched_path']}`")
            preview = get_stitch_preview_frame(st.session_state["stitched_path"])
            if preview is not None:
                st.image(preview, caption="Stitched panorama preview",
                         use_container_width=True)
        return

    # =====================================================================
    # Preview display + approval controls
    # =====================================================================
    candidates = st.session_state["calibration_candidates"]
    idx = st.session_state["current_candidate_idx"]
    current = candidates[idx]

    st.subheader("Stitch Preview")
    if "preview_frame" in st.session_state:
        st.image(st.session_state["preview_frame"],
                 caption=f"Candidate {idx + 1} of {len(candidates)} "
                         f"(frame {current.get('frame_index', 0)})",
                 use_container_width=True)

    # Quality metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Inliers", f"{current['num_inliers']:,}")
    col2.metric("Matches", f"{current['num_matches']:,}")
    col3.metric("Inlier Ratio",
                f"{current.get('inlier_ratio', 0):.1%}")

    # Action buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        can_try = len(candidates) > 1
        if st.button("Try Another", disabled=not can_try):
            next_idx = (idx + 1) % len(candidates)
            st.session_state["current_candidate_idx"] = next_idx
            with st.spinner("Generating preview..."):
                st.session_state["preview_frame"] = generate_preview(
                    left_path, right_path, candidates[next_idx]
                )
            st.rerun()

    with btn_col3:
        if st.button("Reset"):
            for key in ["calibration_candidates", "current_candidate_idx",
                        "preview_frame"]:
                st.session_state.pop(key, None)
            st.rerun()

    # =====================================================================
    # Phase B — Approve + Stitch All Frames
    # =====================================================================
    with btn_col2:
        if st.button("Approve + Stitch All Frames", type="primary"):
            approved = candidates[idx]

            # Save approved calibration
            from datetime import datetime as dt
            cal_date = dt.now().strftime("%Y-%m-%d")
            cal_save_path = os.path.join("calibrations", f"{cal_date}_cal.json")
            os.makedirs("calibrations", exist_ok=True)
            with open(cal_save_path, "w") as f:
                json.dump(approved, f, indent=2)

            progress_bar = st.progress(0, text="Starting stitch...")
            status_text = st.empty()
            stitch_start_time = time.time()

            def on_progress(current_frame, total):
                pct = current_frame / max(total, 1)
                elapsed = time.time() - stitch_start_time
                if current_frame > 0:
                    eta_sec = elapsed / current_frame * (total - current_frame)
                    eta_min = eta_sec / 60
                    if eta_min >= 1:
                        eta_str = f" — ~{eta_min:.0f} min remaining"
                    else:
                        eta_str = f" — ~{eta_sec:.0f}s remaining"
                else:
                    eta_str = ""
                progress_bar.progress(
                    pct,
                    text=f"Stitching frame {current_frame:,} / {total:,}{eta_str}"
                )

            try:
                # Determine frame offset
                if sync_method == "timecode" and tc_offset is not None:
                    fps = left_info["fps"]
                    frame_offset = round(tc_offset * fps)
                else:
                    status_text.text("Syncing audio...")
                    from sync_audio import sync_audio
                    offset_sec = sync_audio(left_path, right_path)
                    frame_offset = round(offset_sec * left_info["fps"])

                status_text.text("Stitching with precomputed remap pipeline...")
                result = stitch_videos(
                    left_path, right_path, output_path,
                    cal_data=approved,
                    frame_offset=frame_offset,
                    progress_callback=on_progress
                )

                progress_bar.progress(1.0, text="Complete!")
                status_text.empty()
                st.success(f"Stitch complete: `{result}`")
                st.session_state["stitched_path"] = result

                # Clear calibration state
                for key in ["calibration_candidates", "current_candidate_idx",
                            "preview_frame"]:
                    st.session_state.pop(key, None)

                # Preview thumbnail of output
                st.subheader("Stitch Result")
                preview = get_stitch_preview_frame(result, frame_index=0)
                if preview is not None:
                    st.image(preview, caption="First stitched frame",
                             use_container_width=True)

            except Exception as e:
                st.error(f"Stitch failed: {e}")


main()
