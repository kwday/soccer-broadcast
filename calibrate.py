"""
calibrate.py — Per-game stitch calibration.

Extracts a single frame from each camera video (or accepts image files directly),
detects matching features in the overlap region using SIFT, computes a homography
matrix with RANSAC, and saves the calibration to a per-game JSON file.

Usage:
    python calibrate.py --left left.mp4 --right right.mp4 [--date 2026-03-15]
    python calibrate.py --left left.jpg --right right.jpg [--date 2026-03-15]
"""

import argparse
import json
import os
import sys
from datetime import date

import cv2
import numpy as np


def extract_frame(source_path: str, frame_index: int = 0) -> np.ndarray:
    """Extract a single frame from a video file, or load an image file directly."""
    ext = os.path.splitext(source_path)[1].lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if ext in image_exts:
        img = cv2.imread(source_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source_path}")
        return img

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source_path}")

    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index} from {source_path}")

    return frame


def detect_and_match(img_left: np.ndarray, img_right: np.ndarray,
                     overlap_fraction: float = 0.35,
                     min_matches: int = 10):
    """
    Detect SIFT features in the overlap regions of both images, match them,
    and return matched keypoints.

    Args:
        img_left: Left camera frame (BGR).
        img_right: Right camera frame (BGR).
        overlap_fraction: Fraction of each image to treat as overlap region.
        min_matches: Minimum number of good matches required.

    Returns:
        pts_left: Matched points in left image (N, 2).
        pts_right: Matched points in right image (N, 2).
    """
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    # Overlap region: right portion of left image, left portion of right image
    overlap_left_start = int(w_left * (1 - overlap_fraction))
    overlap_right_end = int(w_right * overlap_fraction)

    roi_left = img_left[:, overlap_left_start:]
    roi_right = img_right[:, :overlap_right_end]

    # Convert to grayscale
    gray_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY)

    # SIFT feature detection
    sift = cv2.SIFT_create()
    kp_left, desc_left = sift.detectAndCompute(gray_left, None)
    kp_right, desc_right = sift.detectAndCompute(gray_right, None)

    if desc_left is None or desc_right is None:
        raise RuntimeError("Could not detect features in overlap region")

    # BFMatcher with ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(desc_left, desc_right, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < min_matches:
        raise RuntimeError(
            f"Only {len(good_matches)} good matches found (need {min_matches}). "
            "Check overlap region or image quality."
        )

    # Extract matched point coordinates in full image space
    pts_left = np.float32([
        [kp_left[m.queryIdx].pt[0] + overlap_left_start,
         kp_left[m.queryIdx].pt[1]]
        for m in good_matches
    ])
    pts_right = np.float32([
        [kp_right[m.trainIdx].pt[0],
         kp_right[m.trainIdx].pt[1]]
        for m in good_matches
    ])

    return pts_left, pts_right


def compute_homography(pts_left: np.ndarray, pts_right: np.ndarray,
                       ransac_thresh: float = 5.0):
    """
    Compute homography mapping right image into left image's coordinate space.

    Returns:
        H: 3x3 homography matrix.
        mask: Inlier mask from RANSAC.
    """
    H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError("Homography computation failed")
    return H, mask


def compute_canvas_and_blend(img_left: np.ndarray, img_right: np.ndarray,
                             H: np.ndarray):
    """
    Compute the output canvas dimensions and blend region coordinates.

    Returns:
        canvas_width: Width of the stitched panorama.
        canvas_height: Height of the stitched panorama.
        blend_x_start: X coordinate where blending begins (right edge of left-only region).
        blend_x_end: X coordinate where blending ends (left edge of right-only region).
    """
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    # Warp the corners of the right image to find its extent in left image space
    corners_right = np.float32([
        [0, 0], [w_right, 0], [w_right, h_right], [0, h_right]
    ]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_right, H)

    # Canvas bounds
    all_corners = np.concatenate([
        np.float32([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]]).reshape(-1, 1, 2),
        warped_corners
    ])

    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    # Blend region: where left and warped-right overlap
    warped_x_min = int(np.floor(warped_corners[:, 0, 0].min()))
    warped_x_max = int(np.ceil(warped_corners[:, 0, 0].max()))

    blend_x_start = max(warped_x_min - x_min, 0)
    blend_x_end = min(w_left - x_min, warped_x_max - x_min)

    return canvas_width, canvas_height, blend_x_start, blend_x_end, x_min, y_min


def save_calibration(output_path: str, H: np.ndarray,
                     canvas_width: int, canvas_height: int,
                     blend_x_start: int, blend_x_end: int,
                     offset_x: int, offset_y: int,
                     num_matches: int, num_inliers: int,
                     left_shape: tuple, right_shape: tuple):
    """Save calibration data to a JSON file."""
    cal_data = {
        "homography": H.tolist(),
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "blend_x_start": blend_x_start,
        "blend_x_end": blend_x_end,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "left_resolution": list(left_shape[:2]),
        "right_resolution": list(right_shape[:2]),
        "timecode_offset": 0.0
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cal_data, f, indent=2)

    return cal_data


def calibrate(left_path: str, right_path: str,
              cal_date: str = None,
              output_dir: str = "calibrations",
              frame_index: int = 30,
              overlap_fraction: float = 0.35) -> dict:
    """
    Run the full calibration pipeline.

    Args:
        left_path: Path to left camera video/image.
        right_path: Path to right camera video/image.
        cal_date: Date string for output filename (default: today).
        output_dir: Directory for calibration files.
        frame_index: Which frame to extract from videos (default: 30).
        overlap_fraction: Expected overlap fraction (default: 0.35).

    Returns:
        Calibration data dict.
    """
    if cal_date is None:
        cal_date = date.today().strftime("%Y-%m-%d")

    print(f"Extracting frames...")
    img_left = extract_frame(left_path, frame_index)
    img_right = extract_frame(right_path, frame_index)
    print(f"  Left:  {img_left.shape[1]}x{img_left.shape[0]}")
    print(f"  Right: {img_right.shape[1]}x{img_right.shape[0]}")

    print(f"Detecting features in overlap region ({overlap_fraction*100:.0f}%)...")
    pts_left, pts_right = detect_and_match(img_left, img_right, overlap_fraction)
    print(f"  Found {len(pts_left)} good matches")

    print(f"Computing homography...")
    H, mask = compute_homography(pts_left, pts_right)
    num_inliers = int(mask.sum()) if mask is not None else len(pts_left)
    print(f"  Inliers: {num_inliers}/{len(pts_left)}")

    print(f"Computing canvas dimensions...")
    canvas_w, canvas_h, blend_start, blend_end, off_x, off_y = \
        compute_canvas_and_blend(img_left, img_right, H)
    print(f"  Canvas: {canvas_w}x{canvas_h}")
    print(f"  Blend region: x=[{blend_start}, {blend_end}]")

    output_path = os.path.join(output_dir, f"{cal_date}_cal.json")
    print(f"Saving calibration to {output_path}...")
    cal_data = save_calibration(
        output_path, H, canvas_w, canvas_h,
        blend_start, blend_end, off_x, off_y,
        len(pts_left), num_inliers,
        img_left.shape, img_right.shape
    )

    print("Calibration complete.")
    return cal_data


def calibrate_multi(left_path: str, right_path: str,
                    cal_date: str = None,
                    output_dir: str = "calibrations",
                    overlap_fraction: float = 0.35,
                    num_candidates: int = 4) -> list:
    """
    Try multiple candidate frames and return calibration results sorted
    by inlier count (best first).

    Args:
        left_path: Path to left camera video.
        right_path: Path to right camera video.
        cal_date: Date string for output filename.
        output_dir: Directory for calibration files.
        overlap_fraction: Expected overlap fraction.
        num_candidates: Number of candidate frames to try.

    Returns:
        List of calibration data dicts, sorted by num_inliers descending.
    """
    if cal_date is None:
        cal_date = date.today().strftime("%Y-%m-%d")

    # Determine frame count from left video
    cap = cv2.VideoCapture(left_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {left_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 1:
        # Image file or single-frame video: fall back to single calibration
        result = calibrate(left_path, right_path, cal_date, output_dir, 0,
                           overlap_fraction)
        return [result]

    # Candidate frames at 10%, 25%, 50%, 75% of duration
    percentages = [0.10, 0.25, 0.50, 0.75]
    candidate_indices = [min(int(total_frames * p), total_frames - 1)
                         for p in percentages[:num_candidates]]

    results = []
    for idx in candidate_indices:
        try:
            img_left = extract_frame(left_path, idx)
            img_right = extract_frame(right_path, idx)

            pts_left, pts_right = detect_and_match(
                img_left, img_right, overlap_fraction)
            H, mask = compute_homography(pts_left, pts_right)
            num_inliers = int(mask.sum()) if mask is not None else len(pts_left)
            num_matches = len(pts_left)

            canvas_w, canvas_h, blend_start, blend_end, off_x, off_y = \
                compute_canvas_and_blend(img_left, img_right, H)

            results.append({
                "frame_index": idx,
                "homography": H.tolist(),
                "canvas_width": canvas_w,
                "canvas_height": canvas_h,
                "blend_x_start": blend_start,
                "blend_x_end": blend_end,
                "offset_x": off_x,
                "offset_y": off_y,
                "num_matches": num_matches,
                "num_inliers": num_inliers,
                "inlier_ratio": num_inliers / max(num_matches, 1),
                "left_resolution": list(img_left.shape[:2]),
                "right_resolution": list(img_right.shape[:2]),
                "timecode_offset": 0.0,
            })
        except RuntimeError as e:
            print(f"  Candidate frame {idx} failed: {e}")
            continue

    if not results:
        raise RuntimeError("All candidate frames failed calibration")

    # Sort by inlier count (best first)
    results.sort(key=lambda r: r["num_inliers"], reverse=True)

    # Save best result to disk
    best = results[0]
    output_path = os.path.join(output_dir, f"{cal_date}_cal.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Per-game stitch calibration")
    parser.add_argument("--left", required=True, help="Left camera video or image")
    parser.add_argument("--right", required=True, help="Right camera video or image")
    parser.add_argument("--date", default=None, help="Match date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="calibrations", help="Output directory")
    parser.add_argument("--frame", type=int, default=30, help="Frame index to extract")
    parser.add_argument("--overlap", type=float, default=0.35, help="Overlap fraction")
    args = parser.parse_args()

    calibrate(args.left, args.right, args.date, args.output_dir, args.frame, args.overlap)


if __name__ == "__main__":
    main()
