"""
Tests for sync_audio.py

Creates synthetic WAV files with known offsets and verifies:
1. Audio loading works correctly
2. Cross-correlation detects the correct offset
3. Offset accuracy is within 1 frame at 30fps (~33ms)
"""

import os
import sys

import numpy as np
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sync_audio import load_audio, cross_correlate_offset, sync_audio


SAMPLE_RATE = 16000
FRAME_DURATION = 1.0 / 30  # ~33.3ms per frame at 30fps


def make_test_wavs(tmp_path, offset_seconds: float, duration: float = 10.0,
                   sample_rate: int = SAMPLE_RATE):
    """
    Create two WAV files where the right channel is offset by a known amount.

    The signal is a mix of tones and noise to simulate ambient crowd sound.
    """
    rng = np.random.RandomState(42)
    total_samples = int((duration + abs(offset_seconds) + 1) * sample_rate)

    # Create a "crowd noise" signal: pink noise + occasional bursts
    t = np.arange(total_samples) / sample_rate
    signal = rng.randn(total_samples).astype(np.float32) * 0.3

    # Add some tonal components (whistle-like)
    signal += 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    signal += 0.15 * np.sin(2 * np.pi * 880 * t).astype(np.float32)

    # Add random bursts (like crowd cheering)
    for _ in range(5):
        start = rng.randint(0, total_samples - sample_rate)
        burst_len = rng.randint(sample_rate // 4, sample_rate)
        signal[start:start + burst_len] += rng.randn(burst_len).astype(np.float32) * 0.5

    # Clip to safe range
    signal = np.clip(signal, -1.0, 1.0)

    # Create left and right by slicing with offset
    offset_samples = int(offset_seconds * sample_rate)
    num_samples = int(duration * sample_rate)

    if offset_samples >= 0:
        left_signal = signal[:num_samples]
        right_signal = signal[offset_samples:offset_samples + num_samples]
    else:
        right_signal = signal[:num_samples]
        left_signal = signal[-offset_samples:-offset_samples + num_samples]

    # Convert to int16 WAV
    left_int16 = (left_signal * 32767).astype(np.int16)
    right_int16 = (right_signal * 32767).astype(np.int16)

    left_path = str(tmp_path / "left.wav")
    right_path = str(tmp_path / "right.wav")

    wavfile.write(left_path, sample_rate, left_int16)
    wavfile.write(right_path, sample_rate, right_int16)

    return left_path, right_path


class TestLoadAudio:
    def test_loads_int16_wav(self, tmp_path):
        data = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        path = str(tmp_path / "test.wav")
        wavfile.write(path, 16000, data)

        sr, audio = load_audio(path)
        assert sr == 16000
        assert audio.dtype == np.float32
        assert len(audio) == 4
        assert abs(audio[0]) < 0.001
        assert abs(audio[1] - 0.5) < 0.01

    def test_loads_mono(self, tmp_path):
        data = np.zeros(100, dtype=np.int16)
        path = str(tmp_path / "mono.wav")
        wavfile.write(path, 16000, data)

        sr, audio = load_audio(path)
        assert audio.ndim == 1


class TestCrossCorrelation:
    def test_zero_offset(self, tmp_path):
        left_path, right_path = make_test_wavs(tmp_path, offset_seconds=0.0)
        _, left = load_audio(left_path)
        _, right = load_audio(right_path)
        offset = cross_correlate_offset(left, right, SAMPLE_RATE)
        assert abs(offset) < FRAME_DURATION, f"Expected ~0s, got {offset:.4f}s"

    def test_positive_offset_half_second(self, tmp_path):
        """Right camera started 0.5s earlier (right leads)."""
        left_path, right_path = make_test_wavs(tmp_path, offset_seconds=0.5)
        _, left = load_audio(left_path)
        _, right = load_audio(right_path)
        offset = cross_correlate_offset(left, right, SAMPLE_RATE)
        assert abs(offset - 0.5) < FRAME_DURATION, \
            f"Expected ~0.5s, got {offset:.4f}s"

    def test_negative_offset(self, tmp_path):
        """Right camera started 0.3s later (right lags)."""
        left_path, right_path = make_test_wavs(tmp_path, offset_seconds=-0.3)
        _, left = load_audio(left_path)
        _, right = load_audio(right_path)
        offset = cross_correlate_offset(left, right, SAMPLE_RATE)
        assert abs(offset - (-0.3)) < FRAME_DURATION, \
            f"Expected ~-0.3s, got {offset:.4f}s"

    def test_large_offset(self, tmp_path):
        """Test with a 2-second offset."""
        left_path, right_path = make_test_wavs(tmp_path, offset_seconds=2.0,
                                                duration=15.0)
        _, left = load_audio(left_path)
        _, right = load_audio(right_path)
        offset = cross_correlate_offset(left, right, SAMPLE_RATE)
        assert abs(offset - 2.0) < FRAME_DURATION, \
            f"Expected ~2.0s, got {offset:.4f}s"


class TestSyncAudioEndToEnd:
    def test_sync_from_wav_files(self, tmp_path):
        """Full pipeline with WAV file inputs."""
        left_path, right_path = make_test_wavs(tmp_path, offset_seconds=0.75)
        offset = sync_audio(left_path, right_path, sample_rate=SAMPLE_RATE)
        assert abs(offset - 0.75) < FRAME_DURATION, \
            f"Expected ~0.75s, got {offset:.4f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
