"""
smoother.py — Input smoothing utilities.

Provides exponential moving average smoothing for joystick input and
smooth snap-to-position animations for the crop window.
"""

import time


class ExponentialSmoother:
    """
    Exponential moving average smoother for continuous input values.

    output = alpha * input + (1 - alpha) * prev_output

    A higher alpha means less smoothing (more responsive).
    A lower alpha means more smoothing (more sluggish).
    """

    def __init__(self, alpha: float = 0.15, initial: float = 0.0):
        """
        Args:
            alpha: Smoothing factor (0 = frozen, 1 = no smoothing).
            initial: Initial smoothed value.
        """
        self.alpha = max(0.0, min(1.0, alpha))
        self.value = initial

    def update(self, raw_input: float) -> float:
        """Apply smoothing to a new input value and return the smoothed output."""
        self.value = self.alpha * raw_input + (1 - self.alpha) * self.value
        return self.value

    def reset(self, value: float = 0.0):
        """Reset the smoother to a specific value."""
        self.value = value


class SnapAnimator:
    """
    Smooth animation from current position to a target position.

    Uses exponential easing for natural-feeling camera snap movements.
    The animation completes in approximately `duration` seconds.
    """

    def __init__(self, duration: float = 0.5):
        """
        Args:
            duration: Approximate animation duration in seconds.
        """
        self.duration = duration
        self.active = False
        self.start_x = 0.0
        self.start_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.start_time = 0.0

    def start(self, from_x: float, from_y: float, to_x: float, to_y: float):
        """Begin a new snap animation."""
        self.start_x = from_x
        self.start_y = from_y
        self.target_x = to_x
        self.target_y = to_y
        self.start_time = time.time()
        self.active = True

    def update(self) -> tuple:
        """
        Get the current animated position.

        Returns:
            (x, y, is_complete): Current position and whether animation is done.
        """
        if not self.active:
            return self.target_x, self.target_y, True

        elapsed = time.time() - self.start_time
        t = min(elapsed / self.duration, 1.0)

        # Ease-out cubic: 1 - (1-t)^3
        eased = 1 - (1 - t) ** 3

        x = self.start_x + (self.target_x - self.start_x) * eased
        y = self.start_y + (self.target_y - self.start_y) * eased

        if t >= 1.0:
            self.active = False
            return self.target_x, self.target_y, True

        return x, y, False

    def update_with_progress(self, progress: float) -> tuple:
        """
        Get position at a specific progress value (0.0 to 1.0).
        Useful for frame-based animation without relying on wall clock.

        Args:
            progress: Animation progress from 0.0 to 1.0.

        Returns:
            (x, y): Interpolated position.
        """
        t = max(0.0, min(1.0, progress))
        eased = 1 - (1 - t) ** 3

        x = self.start_x + (self.target_x - self.start_x) * eased
        y = self.start_y + (self.target_y - self.start_y) * eased

        if t >= 1.0:
            self.active = False

        return x, y


class InputSmoother:
    """
    Combined smoother for all interactive viewer inputs.

    Provides smoothed pan, tilt, and zoom values plus snap animation support.
    """

    def __init__(self, smoothing_factor: float = 0.15, snap_duration: float = 0.5):
        self.pan = ExponentialSmoother(alpha=smoothing_factor)
        self.tilt = ExponentialSmoother(alpha=smoothing_factor)
        self.zoom = ExponentialSmoother(alpha=smoothing_factor)
        self.snap = SnapAnimator(duration=snap_duration)

    def smooth_input(self, raw_pan: float, raw_tilt: float,
                     raw_zoom: float) -> tuple:
        """
        Apply smoothing to raw input values.

        Returns:
            (smoothed_pan, smoothed_tilt, smoothed_zoom)
        """
        return (
            self.pan.update(raw_pan),
            self.tilt.update(raw_tilt),
            self.zoom.update(raw_zoom),
        )

    def start_snap(self, from_x: float, from_y: float,
                   to_x: float, to_y: float):
        """Start a snap-to-position animation."""
        self.snap.start(from_x, from_y, to_x, to_y)

    def get_snap_position(self) -> tuple:
        """
        Get the current snap animation position.

        Returns:
            (x, y, is_complete)
        """
        return self.snap.update()

    @property
    def is_snapping(self) -> bool:
        return self.snap.active
