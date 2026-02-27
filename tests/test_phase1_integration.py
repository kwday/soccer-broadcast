"""
Phase 1 Integration Test

Runs the full pipeline: calibrate → sync → stitch on synthetic video pairs.
Verifies end-to-end output is a valid stitched panoramic video.
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibrate import calibrate
from sync_audio import cross_correlate_offset
from stitch import stitch_videos, get_video_info


def make_full_test_pair(tmp_path, num_frames=30, width=400, height=300,
                        overlap_px=120, fps=30.0, audio_offset_seconds=0.5):
    """
    Create a full test scenario:
    - Two video files with known overlap
    - Two audio files (WAV) with known offset
    """
    full_width = 2 * width - overlap_px
    rng = np.random.RandomState(42)

    # Base frame with features
    base_frame = rng.randint(50, 200, (height, full_width, 3), dtype=np.uint8)
    for _ in range(30):
        cx = rng.randint(0, full_width)
        cy = rng.randint(0, height)
        r = rng.randint(10, 30)
        color = tuple(int(c) for c in rng.randint(0, 256, 3))
        cv2.circle(base_frame, (cx, cy), r, color, -1)
    base_frame = cv2.GaussianBlur(base_frame, (5, 5), 1.0)

    left_path = str(tmp_path / "left.mp4")
    right_path = str(tmp_path / "right.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_left = cv2.VideoWriter(left_path, fourcc, fps, (width, height))
    writer_right = cv2.VideoWriter(right_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        noise = rng.randint(-3, 4, base_frame.shape, dtype=np.int16)
        frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer_left.write(frame[:, :width])
        writer_right.write(frame[:, (width - overlap_px):])

    writer_left.release()
    writer_right.release()

    # Create audio files with known offset for sync testing
    sr = 16000
    duration = num_frames / fps + abs(audio_offset_seconds) + 1
    total_samples = int(duration * sr)
    t = np.arange(total_samples) / sr
    signal = (rng.randn(total_samples) * 0.3 +
              0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    signal = np.clip(signal, -1.0, 1.0)

    audio_samples = int((num_frames / fps) * sr)
    offset_samples = int(audio_offset_seconds * sr)

    left_audio = (signal[:audio_samples] * 32767).astype(np.int16)
    right_audio = (signal[offset_samples:offset_samples + audio_samples] * 32767).astype(np.int16)

    left_wav = str(tmp_path / "left.wav")
    right_wav = str(tmp_path / "right.wav")
    wavfile.write(left_wav, sr, left_audio)
    wavfile.write(right_wav, sr, right_audio)

    return {
        "left_video": left_path,
        "right_video": right_path,
        "left_audio": left_wav,
        "right_audio": right_wav,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "overlap_px": overlap_px,
        "full_width": full_width,
        "fps": fps,
        "audio_offset": audio_offset_seconds,
    }


class TestPhase1Integration:
    def test_full_pipeline_calibrate_stitch(self, tmp_path):
        """Full pipeline: calibrate → stitch (no audio sync, zero offset)."""
        data = make_full_test_pair(tmp_path)

        # Step 1: Calibrate
        cal_data = calibrate(
            data["left_video"], data["right_video"],
            cal_date="integration-test",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )
        cal_path = str(tmp_path / "cal" / "integration-test_cal.json")
        assert os.path.exists(cal_path)

        # Step 2: Stitch
        output_path = str(tmp_path / "stitched.mp4")
        stitch_videos(
            data["left_video"], data["right_video"],
            output_path,
            cal_path=cal_path,
            frame_offset=0
        )

        # Verify output
        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["frame_count"] == data["num_frames"]
        assert info["width"] > data["width"]  # Panorama is wider than either input
        assert info["height"] >= data["height"]

    def test_full_pipeline_with_audio_sync(self, tmp_path):
        """Full pipeline: calibrate → audio sync → stitch."""
        data = make_full_test_pair(tmp_path, audio_offset_seconds=0.5)

        # Step 1: Calibrate
        cal_data = calibrate(
            data["left_video"], data["right_video"],
            cal_date="integration-audio",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )
        cal_path = str(tmp_path / "cal" / "integration-audio_cal.json")

        # Step 2: Audio sync
        from sync_audio import load_audio
        _, audio_left = load_audio(data["left_audio"])
        _, audio_right = load_audio(data["right_audio"])
        offset = cross_correlate_offset(audio_left, audio_right, 16000)

        # Verify offset is approximately correct (within 1 frame at 30fps)
        assert abs(offset - data["audio_offset"]) < 1.0 / 30, \
            f"Audio offset {offset:.4f}s != expected {data['audio_offset']:.4f}s"

        # Step 3: Stitch with offset
        frame_offset = round(offset * data["fps"])
        output_path = str(tmp_path / "stitched.mp4")
        stitch_videos(
            data["left_video"], data["right_video"],
            output_path,
            cal_path=cal_path,
            frame_offset=frame_offset
        )

        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["frame_count"] > 0

    def test_calibration_json_integrity(self, tmp_path):
        """Verify calibration file has all required fields for downstream use."""
        data = make_full_test_pair(tmp_path)

        cal_data = calibrate(
            data["left_video"], data["right_video"],
            cal_date="integrity-test",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )

        required_fields = [
            "homography", "canvas_width", "canvas_height",
            "blend_x_start", "blend_x_end", "offset_x", "offset_y",
            "num_matches", "num_inliers", "left_resolution",
            "right_resolution", "timecode_offset"
        ]

        for field in required_fields:
            assert field in cal_data, f"Missing field: {field}"

        # Homography should be 3x3
        H = np.array(cal_data["homography"])
        assert H.shape == (3, 3)

        # Dimensions should be positive
        assert cal_data["canvas_width"] > 0
        assert cal_data["canvas_height"] > 0
        assert cal_data["blend_x_start"] < cal_data["blend_x_end"]

    def test_stitched_output_has_content(self, tmp_path):
        """Verify the stitched video frames actually have image content (not all black)."""
        data = make_full_test_pair(tmp_path, num_frames=5)
        output_path = str(tmp_path / "stitched.mp4")

        cal_data = calibrate(
            data["left_video"], data["right_video"],
            cal_date="content-test",
            output_dir=str(tmp_path / "cal"),
            frame_index=0,
            overlap_fraction=0.4
        )
        cal_path = str(tmp_path / "cal" / "content-test_cal.json")

        stitch_videos(
            data["left_video"], data["right_video"],
            output_path,
            cal_path=cal_path,
            frame_offset=0
        )

        # Read back a frame and check it has content
        cap = cv2.VideoCapture(output_path)
        ret, frame = cap.read()
        cap.release()

        assert ret
        # Frame should have significant non-black content
        non_black = (frame.sum(axis=2) > 10).sum()
        total_pixels = frame.shape[0] * frame.shape[1]
        assert non_black / total_pixels > 0.3, \
            f"Only {non_black/total_pixels*100:.1f}% non-black pixels"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
