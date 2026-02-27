"""
Tests for calibrate.py

Creates synthetic left/right images with known overlap and verifies:
1. Feature detection finds matches
2. Homography is computed correctly
3. JSON calibration file is written with correct structure
"""

import json
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibrate import (
    extract_frame,
    detect_and_match,
    compute_homography,
    compute_canvas_and_blend,
    save_calibration,
    calibrate,
)


def make_test_images(width=800, height=600, overlap_px=250):
    """
    Create two synthetic images with a known overlapping region containing
    recognizable features (random textured pattern with shapes).
    """
    # Create a wide source image with rich texture for feature detection
    full_width = 2 * width - overlap_px
    rng = np.random.RandomState(42)

    # Base: random noise texture (features need texture to detect)
    full_img = rng.randint(0, 256, (height, full_width, 3), dtype=np.uint8)

    # Add some geometric shapes for robust features
    for _ in range(50):
        cx = rng.randint(0, full_width)
        cy = rng.randint(0, height)
        r = rng.randint(10, 50)
        color = tuple(int(c) for c in rng.randint(0, 256, 3))
        cv2.circle(full_img, (cx, cy), r, color, -1)

    for _ in range(30):
        x1, y1 = rng.randint(0, full_width), rng.randint(0, height)
        x2, y2 = x1 + rng.randint(20, 100), y1 + rng.randint(20, 100)
        color = tuple(int(c) for c in rng.randint(0, 256, 3))
        cv2.rectangle(full_img, (x1, y1), (x2, y2), color, -1)

    # Apply Gaussian blur to make features more stable
    full_img = cv2.GaussianBlur(full_img, (5, 5), 1.0)

    # Split into left and right with overlap
    img_left = full_img[:, :width].copy()
    img_right = full_img[:, (width - overlap_px):].copy()

    return img_left, img_right


class TestExtractFrame:
    def test_extract_from_image_file(self, tmp_path):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test.jpg")
        cv2.imwrite(img_path, img)

        result = extract_frame(img_path)
        assert result.shape[0] == 100
        assert result.shape[1] == 200

    def test_extract_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_frame("nonexistent.jpg")


class TestDetectAndMatch:
    def test_finds_matches_in_overlapping_images(self):
        img_left, img_right = make_test_images()
        pts_left, pts_right = detect_and_match(
            img_left, img_right, overlap_fraction=0.4, min_matches=5
        )
        assert len(pts_left) >= 5
        assert len(pts_right) >= 5
        assert pts_left.shape[1] == 2
        assert pts_right.shape[1] == 2

    def test_fails_with_no_overlap(self):
        # Two completely different images with no overlap
        rng = np.random.RandomState(1)
        img_left = rng.randint(0, 128, (200, 300, 3), dtype=np.uint8)
        rng2 = np.random.RandomState(999)
        img_right = rng2.randint(128, 256, (200, 300, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="matches found"):
            detect_and_match(img_left, img_right, overlap_fraction=0.35, min_matches=10)


class TestComputeHomography:
    def test_homography_is_3x3(self):
        img_left, img_right = make_test_images()
        pts_left, pts_right = detect_and_match(
            img_left, img_right, overlap_fraction=0.4, min_matches=5
        )
        H, mask = compute_homography(pts_left, pts_right)
        assert H.shape == (3, 3)
        assert mask is not None

    def test_homography_is_approximately_translation(self):
        """With synthetic images, the homography should be close to a pure translation."""
        img_left, img_right = make_test_images(overlap_px=250)
        pts_left, pts_right = detect_and_match(
            img_left, img_right, overlap_fraction=0.4, min_matches=5
        )
        H, _ = compute_homography(pts_left, pts_right)

        # For images that are just offset, H should be close to:
        # [[1, 0, tx], [0, 1, ty], [0, 0, 1]]
        # The translation should be roughly (width - overlap) = 550 in x
        assert abs(H[0, 0] - 1.0) < 0.15, f"H[0,0] = {H[0,0]}, expected ~1.0"
        assert abs(H[1, 1] - 1.0) < 0.15, f"H[1,1] = {H[1,1]}, expected ~1.0"
        assert H[0, 2] > 400, f"Translation x = {H[0,2]}, expected > 400"


class TestCanvasAndBlend:
    def test_canvas_larger_than_either_image(self):
        img_left, img_right = make_test_images()
        pts_left, pts_right = detect_and_match(
            img_left, img_right, overlap_fraction=0.4, min_matches=5
        )
        H, _ = compute_homography(pts_left, pts_right)
        canvas_w, canvas_h, blend_start, blend_end, _, _ = \
            compute_canvas_and_blend(img_left, img_right, H)

        assert canvas_w > img_left.shape[1]
        assert canvas_h >= img_left.shape[0]
        assert blend_start < blend_end


class TestSaveCalibration:
    def test_writes_valid_json(self, tmp_path):
        H = np.eye(3)
        output_path = str(tmp_path / "test_cal.json")
        cal_data = save_calibration(
            output_path, H, 1600, 600, 500, 800, 0, 0,
            50, 45, (600, 800, 3), (600, 800, 3)
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)

        assert "homography" in loaded
        assert len(loaded["homography"]) == 3
        assert len(loaded["homography"][0]) == 3
        assert loaded["canvas_width"] == 1600
        assert loaded["canvas_height"] == 600
        assert loaded["blend_x_start"] == 500
        assert loaded["blend_x_end"] == 800
        assert loaded["num_matches"] == 50
        assert loaded["num_inliers"] == 45
        assert "timecode_offset" in loaded
        assert "left_resolution" in loaded
        assert "right_resolution" in loaded


class TestFullCalibration:
    def test_end_to_end_with_images(self, tmp_path):
        """Full pipeline: save test images, run calibrate(), verify output."""
        img_left, img_right = make_test_images()

        left_path = str(tmp_path / "left.jpg")
        right_path = str(tmp_path / "right.jpg")
        cv2.imwrite(left_path, img_left)
        cv2.imwrite(right_path, img_right)

        cal_dir = str(tmp_path / "calibrations")
        cal_data = calibrate(
            left_path, right_path,
            cal_date="2026-03-15",
            output_dir=cal_dir,
            frame_index=0,
            overlap_fraction=0.4
        )

        # Verify calibration file exists
        cal_file = os.path.join(cal_dir, "2026-03-15_cal.json")
        assert os.path.exists(cal_file)

        # Verify data structure
        with open(cal_file) as f:
            loaded = json.load(f)

        assert loaded["canvas_width"] > 0
        assert loaded["canvas_height"] > 0
        assert len(loaded["homography"]) == 3
        assert loaded["num_inliers"] > 0
        assert loaded["blend_x_start"] < loaded["blend_x_end"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
