"""
Page 1 — Calibrate + Stitch

File selection, metadata display, sync detection, time estimate,
progress bar, and stitch preview thumbnail.
"""

import os
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stitch import get_video_info, stitch_videos, detect_timecode_offset
from app import render_sidebar, load_config


def format_duration(seconds):
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


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


def main():
    st.set_page_config(page_title="Stitch - Soccer Broadcast", layout="wide")
    render_sidebar()

    st.title("Calibrate + Stitch")
    st.markdown("Select left and right camera files, then calibrate and stitch into a panorama.")

    # File selection
    col1, col2 = st.columns(2)
    with col1:
        left_path = st.text_input("Left Camera Video Path",
                                  value=st.session_state.get("left_video_path", ""),
                                  placeholder="C:\\path\\to\\left.mp4")
    with col2:
        right_path = st.text_input("Right Camera Video Path",
                                   value=st.session_state.get("right_video_path", ""),
                                   placeholder="C:\\path\\to\\right.mp4")

    if left_path:
        st.session_state["left_video_path"] = left_path
    if right_path:
        st.session_state["right_video_path"] = right_path

    # Metadata display
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

    # Sync detection
    if left_info and right_info:
        st.markdown("---")
        st.subheader("Sync Detection")
        sync_method, tc_offset = detect_sync_method(left_path, right_path)

        # Time estimate
        duration_sec = left_info["frame_count"] / left_info["fps"] if left_info["fps"] > 0 else 0
        est_minutes = duration_sec * 0.2 / 60
        st.markdown(f"**Estimated stitch time:** ~{est_minutes:.0f} minutes")

        st.markdown("---")

        # Output path
        config = load_config()
        default_output = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "match_stitched.mp4"
        )
        output_path = st.text_input("Output Path", value=default_output)

        # Stitch button
        if st.button("Calibrate + Stitch", type="primary"):
            progress_bar = st.progress(0, text="Starting...")
            status_text = st.empty()

            def on_progress(current, total):
                pct = current / max(total, 1)
                progress_bar.progress(pct, text=f"Stitching frame {current:,} / {total:,}")

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

                status_text.text("Calibrating...")

                result = stitch_videos(
                    left_path, right_path, output_path,
                    frame_offset=frame_offset,
                    progress_callback=on_progress
                )

                progress_bar.progress(1.0, text="Complete!")
                status_text.empty()
                st.success(f"Stitch complete: `{result}`")
                st.session_state["stitched_path"] = result

                # Preview thumbnail
                st.subheader("Stitch Preview")
                preview = get_stitch_preview_frame(result, frame_index=0)
                if preview is not None:
                    st.image(preview, caption="First stitched frame", use_container_width=True)
                else:
                    st.warning("Could not extract preview frame.")

            except Exception as e:
                st.error(f"Stitch failed: {e}")

    # Show previous stitch result if available
    elif st.session_state.get("stitched_path") and os.path.exists(st.session_state["stitched_path"]):
        st.markdown("---")
        st.success(f"Previous stitch result: `{st.session_state['stitched_path']}`")
        preview = get_stitch_preview_frame(st.session_state["stitched_path"])
        if preview is not None:
            st.image(preview, caption="Stitched panorama preview", use_container_width=True)


main()
