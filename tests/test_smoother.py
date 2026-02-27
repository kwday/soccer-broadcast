"""
Tests for smoother.py

Verifies:
1. Exponential smoother ramps smoothly on step input
2. Snap animator eases between positions
3. No jerky movement in animations
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smoother import ExponentialSmoother, SnapAnimator, InputSmoother


class TestExponentialSmoother:
    def test_initial_value(self):
        s = ExponentialSmoother(alpha=0.5, initial=10.0)
        assert s.value == 10.0

    def test_step_input_ramps_smoothly(self):
        """Feed a step input (0 -> 100) and verify output ramps, not jumps."""
        s = ExponentialSmoother(alpha=0.15, initial=0.0)

        values = []
        for _ in range(30):
            values.append(s.update(100.0))

        # First value should be small (not 100)
        assert values[0] < 20, f"First smoothed value {values[0]} is too large"

        # Values should monotonically increase
        for i in range(1, len(values)):
            assert values[i] >= values[i-1], \
                f"Not monotonic: values[{i}]={values[i]} < values[{i-1}]={values[i-1]}"

        # Should approach target after many iterations
        assert values[-1] > 90, f"Final value {values[-1]} didn't reach near target"

    def test_no_smoothing_at_alpha_1(self):
        """Alpha=1 means no smoothing, output matches input exactly."""
        s = ExponentialSmoother(alpha=1.0, initial=0.0)
        assert s.update(42.0) == 42.0
        assert s.update(0.0) == 0.0

    def test_frozen_at_alpha_0(self):
        """Alpha=0 means fully frozen, output never changes."""
        s = ExponentialSmoother(alpha=0.0, initial=50.0)
        assert s.update(100.0) == 50.0
        assert s.update(0.0) == 50.0

    def test_reset(self):
        s = ExponentialSmoother(alpha=0.5, initial=0.0)
        s.update(100.0)
        s.reset(0.0)
        assert s.value == 0.0

    def test_smoothing_reduces_noise(self):
        """Verify that smoothing reduces the variance of noisy input."""
        import numpy as np
        rng = np.random.RandomState(42)
        s = ExponentialSmoother(alpha=0.15, initial=50.0)

        raw_values = 50.0 + rng.randn(100) * 10
        smoothed_values = [s.update(v) for v in raw_values]

        raw_std = np.std(raw_values)
        smoothed_std = np.std(smoothed_values)

        assert smoothed_std < raw_std, \
            f"Smoothed std ({smoothed_std:.2f}) >= raw std ({raw_std:.2f})"


class TestSnapAnimator:
    def test_snap_starts_at_origin(self):
        snap = SnapAnimator(duration=0.5)
        snap.start(0, 0, 100, 200)
        x, y = snap.update_with_progress(0.0)
        assert x == pytest.approx(0, abs=0.1)
        assert y == pytest.approx(0, abs=0.1)

    def test_snap_ends_at_target(self):
        snap = SnapAnimator(duration=0.5)
        snap.start(0, 0, 100, 200)
        x, y = snap.update_with_progress(1.0)
        assert x == pytest.approx(100, abs=0.1)
        assert y == pytest.approx(200, abs=0.1)

    def test_snap_eases_smoothly(self):
        """Verify the animation eases out (fast start, slow end)."""
        snap = SnapAnimator(duration=0.5)
        snap.start(0, 0, 100, 0)

        positions = []
        for i in range(11):
            t = i / 10.0
            x, _ = snap.update_with_progress(t)
            positions.append(x)

        # Should be monotonically increasing
        for i in range(1, len(positions)):
            assert positions[i] >= positions[i-1]

        # Early progress should cover more distance (ease-out)
        first_half_distance = positions[5] - positions[0]
        second_half_distance = positions[10] - positions[5]
        assert first_half_distance > second_half_distance, \
            "Ease-out should cover more distance in first half"

    def test_snap_deactivates_on_completion(self):
        snap = SnapAnimator(duration=0.5)
        snap.start(0, 0, 100, 100)
        assert snap.active is True

        snap.update_with_progress(1.0)
        assert snap.active is False

    def test_snap_time_based(self):
        """Verify time-based animation completes after duration."""
        snap = SnapAnimator(duration=0.1)
        snap.start(0, 0, 100, 200)

        time.sleep(0.15)

        x, y, done = snap.update()
        assert done is True
        assert x == pytest.approx(100, abs=0.1)
        assert y == pytest.approx(200, abs=0.1)


class TestInputSmoother:
    def test_smooth_input(self):
        smoother = InputSmoother(smoothing_factor=0.5)
        pan, tilt, zoom = smoother.smooth_input(10.0, 5.0, 0.5)

        # With alpha=0.5 and initial=0: output = 0.5 * input
        assert pan == pytest.approx(5.0, abs=0.1)
        assert tilt == pytest.approx(2.5, abs=0.1)
        assert zoom == pytest.approx(0.25, abs=0.01)

    def test_snap_integration(self):
        smoother = InputSmoother()
        smoother.start_snap(0, 0, 100, 200)
        assert smoother.is_snapping is True

    def test_snap_position_after_completion(self):
        smoother = InputSmoother(snap_duration=0.05)
        smoother.start_snap(0, 0, 100, 200)
        time.sleep(0.1)
        x, y, done = smoother.get_snap_position()
        assert done is True
        assert x == pytest.approx(100, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
