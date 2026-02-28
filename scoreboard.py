"""
scoreboard.py — Pillow-based scoreboard renderer.

Renders a broadcast scoreboard overlay matching the OBS HTML overlay design.
Used in both the interactive preview (Stage 2) and the final render (Stage 3).
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class ScoreboardState:
    """Current scoreboard state."""
    home_team: str = "HOME"
    away_team: str = "AWAY"
    home_score: int = 0
    away_score: int = 0
    clock_seconds: int = 0
    half: int = 1
    visible: bool = True
    home_color: str = "#1E5E3A"  # dark green
    away_color: str = "#3A1E1E"  # dark red
    position: str = "bottom"     # "top" or "bottom"
    offset: int = 50             # pixels from edge


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def darken_color(rgb: Tuple[int, int, int], factor: float = 0.7) -> Tuple[int, int, int]:
    """Darken an RGB color by a factor."""
    return tuple(max(0, int(c * factor)) for c in rgb)


def blend_color(rgb: Tuple[int, int, int], overlay_rgb: Tuple[int, int, int],
                alpha: float) -> Tuple[int, int, int]:
    """Blend two colors: result = overlay * alpha + base * (1 - alpha)."""
    return tuple(int(o * alpha + b * (1 - alpha)) for o, b in zip(overlay_rgb, rgb))


class ScoreboardRenderer:
    """Renders the scoreboard overlay using Pillow."""

    def __init__(self, width: int = 1920, height: int = 1080,
                 font_path: str = None, mono_font_path: str = None):
        self.frame_width = width
        self.frame_height = height

        # Bar dimensions
        self.bar_height = 60
        self.bar_radius = 8
        self.min_bar_width = 600

        # Load fonts
        self.font_path = font_path or self._find_font("arial.ttf")
        self.mono_font_path = mono_font_path or self._find_font("cour.ttf")

        self.font_team_name = self._load_font(self.font_path, 18)
        self.font_score = self._load_font(self.font_path, 32)
        self.font_timer = self._load_font(self.mono_font_path, 24)
        self.font_half = self._load_font(self.font_path, 10)
        self.font_logo = self._load_font(self.font_path, 22)

        # Cache
        self._cache_state = None
        self._cache_image = None

    def _find_font(self, name: str) -> str:
        """Search for a font file in common locations."""
        search_paths = [
            os.path.join(os.path.dirname(__file__), "assets", "fonts", name),
            os.path.join("assets", "fonts", name),
            os.path.join("C:/Windows/Fonts", name),
            os.path.join("/usr/share/fonts/truetype", name),
        ]
        for path in search_paths:
            if os.path.exists(path):
                return path
        return name  # Fall back to name, let Pillow try

    def _load_font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        """Load a TTF font, with fallback to default."""
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            return ImageFont.load_default()

    def _measure_text(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        """Measure text dimensions."""
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def render(self, state: ScoreboardState) -> Image.Image:
        """
        Render the scoreboard overlay as a transparent RGBA image.

        Returns:
            PIL Image (RGBA) at frame_width x frame_height with scoreboard drawn.
        """
        if not state.visible:
            return Image.new("RGBA", (self.frame_width, self.frame_height), (0, 0, 0, 0))

        # Check cache
        cache_key = (
            state.home_team, state.away_team, state.home_score, state.away_score,
            state.clock_seconds, state.half, state.visible,
            state.home_color, state.away_color, state.position, state.offset
        )
        if cache_key == self._cache_state and self._cache_image is not None:
            return self._cache_image

        # Create transparent canvas
        img = Image.new("RGBA", (self.frame_width, self.frame_height), (0, 0, 0, 0))

        # Calculate bar width based on team name lengths
        home_name_w, _ = self._measure_text(state.home_team.upper(), self.font_team_name)
        away_name_w, _ = self._measure_text(state.away_team.upper(), self.font_team_name)

        logo_section_w = 45 + 10  # logo + margin
        name_padding = 25  # padding around name
        score_section_w = 60
        timer_section_w = 100

        home_section_w = logo_section_w + home_name_w + name_padding + score_section_w
        away_section_w = score_section_w + away_name_w + name_padding + logo_section_w

        bar_width = max(self.min_bar_width,
                        home_section_w + timer_section_w + away_section_w)

        # Position the bar
        bar_x = (self.frame_width - bar_width) // 2
        if state.position == "top":
            bar_y = state.offset
        else:
            bar_y = self.frame_height - self.bar_height - state.offset

        # Draw bar on a sub-image for rounded corners
        bar_img = Image.new("RGBA", (bar_width, self.bar_height), (0, 0, 0, 0))
        bar_draw = ImageDraw.Draw(bar_img)

        # Bar background (dark gradient approximation)
        bar_draw.rounded_rectangle(
            [0, 0, bar_width - 1, self.bar_height - 1],
            radius=self.bar_radius,
            fill=(38, 38, 38, 242),  # ~#262626 at 95% opacity
            outline=(68, 68, 68, 204),  # border
            width=2
        )

        # === HOME SECTION ===
        home_rgb = hex_to_rgb(state.home_color)
        home_dark = darken_color(home_rgb, 0.7)

        # Home background (team color)
        home_end_x = home_section_w
        bar_draw.rectangle(
            [self.bar_radius, 2, home_end_x, self.bar_height - 3],
            fill=(*home_rgb, 230)
        )

        # Home logo (semi-transparent white box with initial)
        logo_x = 10
        logo_y = (self.bar_height - 45) // 2
        logo_bg = blend_color(home_rgb, (255, 255, 255), 0.15)
        bar_draw.rounded_rectangle(
            [logo_x, logo_y, logo_x + 45, logo_y + 45],
            radius=5,
            fill=(*logo_bg, 255)
        )
        initial = state.home_team[0].upper() if state.home_team else "H"
        iw, ih = self._measure_text(initial, self.font_logo)
        bar_draw.text(
            (logo_x + (45 - iw) // 2, logo_y + (45 - ih) // 2 - 2),
            initial, fill=(255, 255, 255, 255), font=self.font_logo
        )

        # Home team name
        name_x = logo_x + 45 + 10
        name_y = (self.bar_height - 18) // 2 - 2
        # Text shadow
        bar_draw.text((name_x + 2, name_y + 2), state.home_team.upper(),
                      fill=(0, 0, 0, 200), font=self.font_team_name)
        bar_draw.text((name_x, name_y), state.home_team.upper(),
                      fill=(255, 255, 255, 255), font=self.font_team_name)

        # Home score box (darkened team color)
        score_box_x = home_end_x - score_section_w
        score_bg = blend_color(home_rgb, (0, 0, 0), 0.4)
        bar_draw.rectangle(
            [score_box_x, 2, home_end_x, self.bar_height - 3],
            fill=(*score_bg, 255)
        )
        # Score border
        bar_draw.line([(score_box_x, 2), (score_box_x, self.bar_height - 3)],
                      fill=(51, 51, 51, 255), width=2)

        score_text = str(state.home_score)
        sw, sh = self._measure_text(score_text, self.font_score)
        score_x = score_box_x + (score_section_w - sw) // 2
        score_y = (self.bar_height - sh) // 2 - 4
        bar_draw.text((score_x + 2, score_y + 2), score_text,
                      fill=(0, 0, 0, 200), font=self.font_score)
        bar_draw.text((score_x, score_y), score_text,
                      fill=(255, 255, 255, 255), font=self.font_score)

        # === TIMER SECTION ===
        timer_x = home_end_x
        timer_end_x = timer_x + timer_section_w
        bar_draw.rectangle(
            [timer_x, 2, timer_end_x, self.bar_height - 3],
            fill=(50, 50, 50, 230)
        )
        # Timer borders
        bar_draw.line([(timer_x, 2), (timer_x, self.bar_height - 3)],
                      fill=(85, 85, 85, 255), width=2)
        bar_draw.line([(timer_end_x, 2), (timer_end_x, self.bar_height - 3)],
                      fill=(85, 85, 85, 255), width=2)

        # Clock text
        minutes = state.clock_seconds // 60
        seconds = state.clock_seconds % 60
        clock_text = f"{minutes:02d}:{seconds:02d}"
        cw, ch = self._measure_text(clock_text, self.font_timer)
        clock_x = timer_x + (timer_section_w - cw) // 2
        clock_y = (self.bar_height - ch) // 2 - 8
        # Gold timer text with shadow
        bar_draw.text((clock_x + 2, clock_y + 2), clock_text,
                      fill=(0, 0, 0, 200), font=self.font_timer)
        bar_draw.text((clock_x, clock_y), clock_text,
                      fill=(255, 215, 0, 255), font=self.font_timer)  # #FFD700

        # Half indicator
        half_text = "1ST HALF" if state.half == 1 else "2ND HALF"
        hw, hh = self._measure_text(half_text, self.font_half)
        half_x = timer_x + (timer_section_w - hw) // 2
        half_y = clock_y + ch + 4
        bar_draw.text((half_x, half_y), half_text,
                      fill=(170, 170, 170, 255), font=self.font_half)  # #AAA

        # === AWAY SECTION ===
        away_rgb = hex_to_rgb(state.away_color)
        away_start_x = timer_end_x

        # Away background
        bar_draw.rectangle(
            [away_start_x, 2, bar_width - self.bar_radius, self.bar_height - 3],
            fill=(*away_rgb, 230)
        )

        # Away score box
        away_score_end_x = away_start_x + score_section_w
        away_score_bg = blend_color(away_rgb, (0, 0, 0), 0.4)
        bar_draw.rectangle(
            [away_start_x, 2, away_score_end_x, self.bar_height - 3],
            fill=(*away_score_bg, 255)
        )
        bar_draw.line([(away_score_end_x, 2), (away_score_end_x, self.bar_height - 3)],
                      fill=(51, 51, 51, 255), width=2)

        away_score_text = str(state.away_score)
        asw, ash = self._measure_text(away_score_text, self.font_score)
        away_score_x = away_start_x + (score_section_w - asw) // 2
        away_score_y = (self.bar_height - ash) // 2 - 4
        bar_draw.text((away_score_x + 2, away_score_y + 2), away_score_text,
                      fill=(0, 0, 0, 200), font=self.font_score)
        bar_draw.text((away_score_x, away_score_y), away_score_text,
                      fill=(255, 255, 255, 255), font=self.font_score)

        # Away team name
        away_name_x = away_score_end_x + 10
        bar_draw.text((away_name_x + 2, name_y + 2), state.away_team.upper(),
                      fill=(0, 0, 0, 200), font=self.font_team_name)
        bar_draw.text((away_name_x, name_y), state.away_team.upper(),
                      fill=(255, 255, 255, 255), font=self.font_team_name)

        # Away logo
        away_logo_x = bar_width - 45 - 10
        bar_draw.rounded_rectangle(
            [away_logo_x, logo_y, away_logo_x + 45, logo_y + 45],
            radius=5,
            fill=(*blend_color(away_rgb, (255, 255, 255), 0.15), 255)
        )
        away_initial = state.away_team[0].upper() if state.away_team else "A"
        aiw, aih = self._measure_text(away_initial, self.font_logo)
        bar_draw.text(
            (away_logo_x + (45 - aiw) // 2, logo_y + (45 - aih) // 2 - 2),
            away_initial, fill=(255, 255, 255, 255), font=self.font_logo
        )

        # Paste bar onto main canvas
        img.paste(bar_img, (bar_x, bar_y), bar_img)

        # Update cache
        self._cache_state = cache_key
        self._cache_image = img

        return img

    def render_to_bgr(self, state: ScoreboardState) -> 'np.ndarray':
        """Render scoreboard and return as BGR numpy array for OpenCV compositing."""
        import numpy as np
        rgba_img = self.render(state)
        rgba_array = np.array(rgba_img)
        # Convert RGBA to BGRA
        bgra = rgba_array[:, :, [2, 1, 0, 3]]
        return bgra

    def composite_onto_frame(self, frame: 'np.ndarray',
                             state: ScoreboardState) -> 'np.ndarray':
        """
        Composite the scoreboard overlay onto a BGR video frame.

        Args:
            frame: BGR video frame (numpy array, H x W x 3).
            state: Current scoreboard state.

        Returns:
            Frame with scoreboard composited (BGR, same shape).
        """
        import numpy as np

        if not state.visible:
            return frame

        bgra = self.render_to_bgr(state)

        # Resize overlay if frame size doesn't match
        h, w = frame.shape[:2]
        if bgra.shape[0] != h or bgra.shape[1] != w:
            from PIL import Image as PILImage
            rgba_img = self.render(state)
            rgba_img = rgba_img.resize((w, h), PILImage.LANCZOS)
            rgba_array = np.array(rgba_img)
            bgra = rgba_array[:, :, [2, 1, 0, 3]]

        # Find bounding box of non-transparent pixels to avoid full-frame blend
        alpha_full = bgra[:, :, 3]
        rows_with_content = np.any(alpha_full > 0, axis=1)
        cols_with_content = np.any(alpha_full > 0, axis=0)

        if not np.any(rows_with_content):
            return frame

        y0 = int(np.argmax(rows_with_content))
        y1 = int(len(rows_with_content) - np.argmax(rows_with_content[::-1]))
        x0 = int(np.argmax(cols_with_content))
        x1 = int(len(cols_with_content) - np.argmax(cols_with_content[::-1]))

        # Alpha composite only the scoreboard region
        roi_alpha = alpha_full[y0:y1, x0:x1].astype(np.float32) / 255.0
        alpha_3ch = roi_alpha[:, :, np.newaxis]
        roi_bgr = bgra[y0:y1, x0:x1, :3].astype(np.float32)
        roi_frame = frame[y0:y1, x0:x1].astype(np.float32)

        blended = roi_bgr * alpha_3ch + roi_frame * (1 - alpha_3ch)
        result = frame.copy()
        result[y0:y1, x0:x1] = blended.astype(np.uint8)

        return result
