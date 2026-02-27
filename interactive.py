"""
interactive.py — Interactive PTZ viewer.

Loads a stitched panoramic video, displays it with a crop rectangle overlay,
and allows pan/tilt/zoom control via keyboard (and later joystick).

Usage:
    python interactive.py --video stitched.mp4
    python interactive.py --video stitched.mp4 --debug-controller
"""

import argparse
import csv
import os
import sys
import time
from datetime import date

import cv2
import numpy as np

# pygame imported with error suppression for headless environments
try:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class CropState:
    """Tracks the current crop window position and zoom level."""

    def __init__(self, pano_width: int, pano_height: int,
                 output_width: int = 1920, output_height: int = 1080):
        self.pano_width = pano_width
        self.pano_height = pano_height
        self.output_width = output_width
        self.output_height = output_height

        # Zoom: 1.0 = native (crop = output size), 1.2 = 20% upscale, 0 = full pano
        self.zoom = 1.0
        self.min_zoom = 0.0
        self.max_zoom = 1.2

        # Crop center position (in panorama coordinates)
        self.center_x = pano_width / 2.0
        self.center_y = pano_height / 2.0

    @property
    def crop_w(self) -> int:
        if self.zoom == 0:
            return self.pano_width
        return int(self.output_width / self.zoom)

    @property
    def crop_h(self) -> int:
        if self.zoom == 0:
            return self.pano_height
        return int(self.output_height / self.zoom)

    @property
    def crop_x(self) -> int:
        x = int(self.center_x - self.crop_w / 2)
        x = max(0, min(x, self.pano_width - self.crop_w))
        return x

    @property
    def crop_y(self) -> int:
        y = int(self.center_y - self.crop_h / 2)
        y = max(0, min(y, self.pano_height - self.crop_h))
        return y

    def move(self, dx: float, dy: float):
        """Move crop center by (dx, dy) pixels."""
        self.center_x = np.clip(
            self.center_x + dx,
            self.crop_w / 2,
            self.pano_width - self.crop_w / 2
        )
        self.center_y = np.clip(
            self.center_y + dy,
            self.crop_h / 2,
            self.pano_height - self.crop_h / 2
        )

    def adjust_zoom(self, delta: float):
        """Adjust zoom level by delta."""
        if self.zoom == 0 and delta > 0:
            self.zoom = 0.5  # Jump from full pano to 0.5x
        else:
            self.zoom = np.clip(self.zoom + delta, self.min_zoom, self.max_zoom)
            if self.zoom < 0.1 and delta < 0:
                self.zoom = 0.0  # Snap to full pano


class InteractiveViewer:
    """Main interactive viewer application."""

    def __init__(self, video_path: str, config: dict = None):
        self.video_path = video_path
        self.config = config or {}

        # Video
        self.cap = None
        self.pano_width = 0
        self.pano_height = 0
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0

        # State
        self.crop = None
        self.running = False

        # Display
        self.screen = None
        self.window_width = 1280
        self.window_height = 720

        # Keyboard control speeds
        self.pan_speed = self.config.get("pan_speed", 15.0)
        self.tilt_speed = self.config.get("tilt_speed", 10.0)
        self.zoom_speed = self.config.get("zoom_speed", 0.02)

        # Joystick (initialized later)
        self.joystick = None

        # Scoreboard state
        self.home_score = 0
        self.away_score = 0
        self.clock_running = False
        self.clock_seconds = 0.0
        self.half = 1
        self.scoreboard_visible = True
        self.scoreboard_renderer = None

        # Match info
        self.home_team = self.config.get("home_team", "HOME")
        self.away_team = self.config.get("away_team", "AWAY")
        self.home_color = self.config.get("home_color", "#1E3A5F")
        self.away_color = self.config.get("away_color", "#8B0000")

        # Logging
        self.log_rows = []

    def open_video(self):
        """Open the video file and read metadata."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        self.pano_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.pano_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.crop = CropState(self.pano_width, self.pano_height)

    def init_display(self):
        """Initialize pygame display."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is not available")

        pygame.init()
        pygame.display.set_caption("Soccer Broadcast - Interactive PTZ")

        # Layout: panorama overview on top, crop preview on bottom
        self.window_width = 1280
        self.window_height = 720
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )

    def init_joystick(self):
        """Try to initialize a joystick/controller."""
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Controller: {self.joystick.get_name()}")
        else:
            print("No controller detected. Using keyboard only.")

    def handle_keyboard(self, keys):
        """Handle keyboard input for crop movement."""
        # Arrow keys for pan/tilt
        if keys[pygame.K_LEFT]:
            self.crop.move(-self.pan_speed, 0)
        if keys[pygame.K_RIGHT]:
            self.crop.move(self.pan_speed, 0)
        if keys[pygame.K_UP]:
            self.crop.move(0, -self.tilt_speed)
        if keys[pygame.K_DOWN]:
            self.crop.move(0, self.tilt_speed)

        # +/- for zoom
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            self.crop.adjust_zoom(self.zoom_speed)
        if keys[pygame.K_MINUS]:
            self.crop.adjust_zoom(-self.zoom_speed)

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                # Scoreboard controls
                elif event.key == pygame.K_SPACE:
                    self.clock_running = not self.clock_running
                elif event.key == pygame.K_q:
                    self.home_score += 1
                elif event.key == pygame.K_a:
                    self.home_score = max(0, self.home_score - 1)
                elif event.key == pygame.K_p:
                    self.away_score += 1
                elif event.key == pygame.K_l:
                    self.away_score = max(0, self.away_score - 1)
                elif event.key == pygame.K_r:
                    self.clock_seconds = 0.0
                elif event.key == pygame.K_h:
                    self.half = 2 if self.half == 1 else 1
                    self.clock_seconds = 0.0
                    self.clock_running = False
                elif event.key == pygame.K_t:
                    self.scoreboard_visible = not self.scoreboard_visible

    def draw_frame(self, frame: np.ndarray):
        """Draw the panorama overview and crop preview."""
        if self.screen is None:
            return

        self.screen.fill((0, 0, 0))

        # Top section: downscaled panorama with crop rectangle
        pano_display_w = self.window_width
        pano_display_h = int(self.pano_height * pano_display_w / self.pano_width)
        pano_display_h = min(pano_display_h, self.window_height // 2)

        scale_x = pano_display_w / self.pano_width
        scale_y = pano_display_h / self.pano_height

        pano_small = cv2.resize(frame, (pano_display_w, pano_display_h))

        # Draw crop rectangle on panorama
        rect_x = int(self.crop.crop_x * scale_x)
        rect_y = int(self.crop.crop_y * scale_y)
        rect_w = int(self.crop.crop_w * scale_x)
        rect_h = int(self.crop.crop_h * scale_y)
        cv2.rectangle(pano_small, (rect_x, rect_y),
                       (rect_x + rect_w, rect_y + rect_h),
                       (0, 255, 0), 2)

        # Convert BGR to RGB for pygame
        pano_rgb = cv2.cvtColor(pano_small, cv2.COLOR_BGR2RGB)
        pano_surface = pygame.image.frombuffer(
            pano_rgb.tobytes(), (pano_display_w, pano_display_h), "RGB"
        )
        self.screen.blit(pano_surface, (0, 0))

        # Bottom section: crop preview
        crop_x = self.crop.crop_x
        crop_y = self.crop.crop_y
        crop_w = self.crop.crop_w
        crop_h = self.crop.crop_h

        # Ensure we don't go out of bounds
        crop_x = max(0, min(crop_x, self.pano_width - crop_w))
        crop_y = max(0, min(crop_y, self.pano_height - crop_h))

        cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        if cropped.size > 0:
            # Scale to fit bottom half of window
            preview_w = self.window_width // 2
            preview_h = self.window_height - pano_display_h
            if preview_h > 0:
                # Maintain 16:9 aspect ratio
                target_h = int(preview_w * 9 / 16)
                if target_h > preview_h:
                    target_h = preview_h
                    preview_w = int(target_h * 16 / 9)

                preview = cv2.resize(cropped, (preview_w, target_h))
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                preview_surface = pygame.image.frombuffer(
                    preview_rgb.tobytes(), (preview_w, target_h), "RGB"
                )
                # Center in bottom half
                px = (self.window_width - preview_w) // 2
                py = pano_display_h + (preview_h - target_h) // 2
                self.screen.blit(preview_surface, (px, py))

        # Draw info text
        font = pygame.font.SysFont("monospace", 14)
        info_text = (
            f"Frame: {self.current_frame}/{self.total_frames}  "
            f"Crop: ({self.crop.crop_x},{self.crop.crop_y}) {self.crop.crop_w}x{self.crop.crop_h}  "
            f"Zoom: {self.crop.zoom:.2f}x  "
            f"Score: {self.home_score}-{self.away_score}  "
            f"Clock: {int(self.clock_seconds//60):02d}:{int(self.clock_seconds%60):02d}  "
            f"Half: {self.half}"
        )
        text_surface = font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, self.window_height - 20))

        pygame.display.flip()

    def log_frame(self, timestamp: float):
        """Log the current frame state."""
        self.log_rows.append({
            "frame": self.current_frame,
            "timestamp": f"{timestamp:.3f}",
            "crop_x": self.crop.crop_x,
            "crop_y": self.crop.crop_y,
            "crop_w": self.crop.crop_w,
            "crop_h": self.crop.crop_h,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "clock_running": str(self.clock_running).lower(),
            "clock_seconds": int(self.clock_seconds),
            "half": self.half,
            "scoreboard_visible": str(self.scoreboard_visible).lower(),
        })

    def save_log(self, output_path: str = None):
        """Save the session log to CSV."""
        if not self.log_rows:
            return

        if output_path is None:
            os.makedirs("logs", exist_ok=True)
            output_path = os.path.join(
                "logs", f"{date.today().strftime('%Y-%m-%d')}_match.csv"
            )

        fieldnames = [
            "frame", "timestamp", "crop_x", "crop_y", "crop_w", "crop_h",
            "home_score", "away_score", "clock_running", "clock_seconds",
            "half", "scoreboard_visible"
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log_rows)

        print(f"Log saved: {output_path} ({len(self.log_rows)} frames)")
        return output_path

    def run(self, headless: bool = False, max_frames: int = None):
        """
        Run the interactive viewer.

        Args:
            headless: If True, run without display (for testing).
            max_frames: Stop after this many frames (for testing).
        """
        self.open_video()

        if not headless:
            self.init_display()
            self.init_joystick()

        self.running = True
        frame_duration = 1.0 / self.fps
        clock_tick_rate = 1.0 / self.fps

        while self.running:
            frame_start = time.time()

            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = self.current_frame / self.fps

            # Update clock
            if self.clock_running:
                self.clock_seconds += clock_tick_rate

            if not headless:
                # Handle input
                self.handle_events()
                keys = pygame.key.get_pressed()
                self.handle_keyboard(keys)

                # Draw
                self.draw_frame(frame)

            # Log
            self.log_frame(timestamp)

            self.current_frame += 1

            if max_frames and self.current_frame >= max_frames:
                break

            # Frame timing
            if not headless:
                elapsed = time.time() - frame_start
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Cleanup
        self.cap.release()
        if not headless and PYGAME_AVAILABLE:
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Interactive PTZ viewer")
    parser.add_argument("--video", required=True, help="Stitched panoramic video")
    parser.add_argument("--debug-controller", action="store_true",
                        help="Print controller axis/button values")
    parser.add_argument("--log-output", default=None, help="Log file path")
    args = parser.parse_args()

    viewer = InteractiveViewer(args.video)
    viewer.run()
    viewer.save_log(args.log_output)


if __name__ == "__main__":
    main()
