"""
sync_audio.py — Fallback audio cross-correlation sync.

When GoPro timecode sync is not available, this module extracts audio from
both video files and uses cross-correlation to determine the time offset
between them.

Usage:
    python sync_audio.py --left left.mp4 --right right.mp4
    python sync_audio.py --left left.wav --right right.wav
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve


def extract_audio(video_path: str, output_wav: str, sample_rate: int = 16000):
    """
    Extract mono audio from a video file using ffmpeg.

    Args:
        video_path: Path to the video file.
        output_wav: Path for the output WAV file.
        sample_rate: Target sample rate (default: 16000 Hz).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",              # no video
        "-ac", "1",         # mono
        "-ar", str(sample_rate),
        "-f", "wav",
        output_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")


def load_audio(path: str) -> tuple:
    """
    Load audio from a WAV file.

    Returns:
        sample_rate: Sample rate in Hz.
        data: Audio data as float32 numpy array, normalized to [-1, 1].
    """
    sr, data = wavfile.read(path)

    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float64:
        data = data.astype(np.float32)

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    return sr, data


def cross_correlate_offset(audio_left: np.ndarray, audio_right: np.ndarray,
                           sample_rate: int, max_offset_seconds: float = 30.0) -> float:
    """
    Compute the time offset between two audio signals using cross-correlation.

    A positive result means audio_right is ahead (started earlier) by that many seconds.
    A negative result means audio_right is behind (started later).

    Args:
        audio_left: Left channel audio (float32).
        audio_right: Right channel audio (float32).
        sample_rate: Sample rate of both signals.
        max_offset_seconds: Maximum offset to search (limits computation).

    Returns:
        Offset in seconds (positive = right leads, negative = right lags).
    """
    max_offset_samples = int(max_offset_seconds * sample_rate)

    # Truncate to reasonable length for cross-correlation (use first 60 seconds)
    max_samples = min(60 * sample_rate, len(audio_left), len(audio_right))
    a = audio_left[:max_samples]
    b = audio_right[:max_samples]

    # Normalize
    a = (a - a.mean()) / (a.std() + 1e-10)
    b = (b - b.mean()) / (b.std() + 1e-10)

    # Cross-correlate using FFT
    correlation = fftconvolve(a, b[::-1], mode="full")

    # The center of the correlation output corresponds to zero lag
    center = len(a) - 1

    # Limit search range
    search_start = max(0, center - max_offset_samples)
    search_end = min(len(correlation), center + max_offset_samples + 1)

    search_region = correlation[search_start:search_end]
    peak_index = np.argmax(search_region) + search_start

    # Convert to offset in samples (positive means right leads)
    offset_samples = peak_index - center
    offset_seconds = offset_samples / sample_rate

    return offset_seconds


def sync_audio(left_path: str, right_path: str,
               sample_rate: int = 16000,
               max_offset: float = 30.0) -> float:
    """
    Compute the audio sync offset between two video or audio files.

    Args:
        left_path: Path to left video/audio file.
        right_path: Path to right video/audio file.
        sample_rate: Sample rate for audio extraction.
        max_offset: Maximum offset to search in seconds.

    Returns:
        Offset in seconds.
    """
    ext_left = os.path.splitext(left_path)[1].lower()
    ext_right = os.path.splitext(right_path)[1].lower()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract or load audio
        if ext_left == ".wav":
            wav_left = left_path
        else:
            wav_left = os.path.join(tmpdir, "left.wav")
            print(f"Extracting audio from left: {left_path}")
            extract_audio(left_path, wav_left, sample_rate)

        if ext_right == ".wav":
            wav_right = right_path
        else:
            wav_right = os.path.join(tmpdir, "right.wav")
            print(f"Extracting audio from right: {right_path}")
            extract_audio(right_path, wav_right, sample_rate)

        print("Loading audio...")
        sr_left, audio_left = load_audio(wav_left)
        sr_right, audio_right = load_audio(wav_right)

        # Resample if needed (simple case: just use what we extracted at target rate)
        if sr_left != sr_right:
            raise RuntimeError(
                f"Sample rate mismatch: left={sr_left}, right={sr_right}. "
                "Extract both at the same rate."
            )

        print(f"Cross-correlating ({len(audio_left)/sr_left:.1f}s vs {len(audio_right)/sr_right:.1f}s)...")
        offset = cross_correlate_offset(audio_left, audio_right, sr_left, max_offset)

        print(f"Detected offset: {offset:+.4f} seconds ({offset * sr_left:+.0f} samples)")
        return offset


def main():
    parser = argparse.ArgumentParser(description="Audio cross-correlation sync")
    parser.add_argument("--left", required=True, help="Left camera video or WAV")
    parser.add_argument("--right", required=True, help="Right camera video or WAV")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (Hz)")
    parser.add_argument("--max-offset", type=float, default=30.0, help="Max offset (seconds)")
    args = parser.parse_args()

    offset = sync_audio(args.left, args.right, args.sample_rate, args.max_offset)
    print(f"\nResult: {offset:+.4f} seconds")


if __name__ == "__main__":
    main()
