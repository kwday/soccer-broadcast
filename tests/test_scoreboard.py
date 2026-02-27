"""
Tests for scoreboard.py

Verifies:
1. Renders scoreboard to PNG with sample state
2. Tests all states: 0-0, 3-2, half change, visibility toggle
3. Composite onto video frame works
4. Cache works (same state returns same image)
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoreboard import ScoreboardRenderer, ScoreboardState, hex_to_rgb, darken_color


class TestColorUtils:
    def test_hex_to_rgb(self):
        assert hex_to_rgb("#FF0000") == (255, 0, 0)
        assert hex_to_rgb("#00FF00") == (0, 255, 0)
        assert hex_to_rgb("#1E5E3A") == (30, 94, 58)
        assert hex_to_rgb("1E5E3A") == (30, 94, 58)  # without #

    def test_darken_color(self):
        result = darken_color((100, 200, 50), 0.5)
        assert result == (50, 100, 25)


class TestScoreboardState:
    def test_defaults(self):
        state = ScoreboardState()
        assert state.home_team == "HOME"
        assert state.away_team == "AWAY"
        assert state.home_score == 0
        assert state.away_score == 0
        assert state.clock_seconds == 0
        assert state.half == 1
        assert state.visible is True


class TestScoreboardRenderer:
    @pytest.fixture
    def renderer(self):
        return ScoreboardRenderer(1920, 1080)

    def test_render_returns_rgba_image(self, renderer):
        state = ScoreboardState()
        img = renderer.render(state)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGBA"
        assert img.size == (1920, 1080)

    def test_render_has_non_transparent_content(self, renderer):
        state = ScoreboardState()
        img = renderer.render(state)
        arr = np.array(img)
        # Alpha channel should have non-zero values (the scoreboard bar)
        assert arr[:, :, 3].sum() > 0

    def test_render_zero_zero(self, renderer):
        """Test 0-0 score state."""
        state = ScoreboardState(home_score=0, away_score=0)
        img = renderer.render(state)
        assert img is not None
        assert img.size == (1920, 1080)

    def test_render_three_two(self, renderer):
        """Test 3-2 score state."""
        state = ScoreboardState(home_score=3, away_score=2)
        img = renderer.render(state)
        assert img is not None

    def test_render_half_change(self, renderer):
        """Test both halves render differently."""
        state1 = ScoreboardState(half=1)
        state2 = ScoreboardState(half=2)
        img1 = renderer.render(state1)
        img2 = renderer.render(state2)

        arr1 = np.array(img1)
        arr2 = np.array(img2)
        # Images should differ (different half text)
        assert not np.array_equal(arr1, arr2)

    def test_render_invisible(self, renderer):
        """Test invisible state returns fully transparent image."""
        state = ScoreboardState(visible=False)
        img = renderer.render(state)
        arr = np.array(img)
        assert arr[:, :, 3].sum() == 0

    def test_render_visibility_toggle(self, renderer):
        """Toggle visibility and verify output changes."""
        visible_state = ScoreboardState(visible=True)
        hidden_state = ScoreboardState(visible=False)

        visible_img = renderer.render(visible_state)
        hidden_img = renderer.render(hidden_state)

        v_arr = np.array(visible_img)
        h_arr = np.array(hidden_img)

        assert v_arr[:, :, 3].sum() > 0
        assert h_arr[:, :, 3].sum() == 0

    def test_render_custom_team_names(self, renderer):
        state = ScoreboardState(home_team="Eagles", away_team="Hawks")
        img = renderer.render(state)
        assert img is not None

    def test_render_custom_colors(self, renderer):
        state = ScoreboardState(home_color="#0000FF", away_color="#FF0000")
        img = renderer.render(state)
        assert img is not None

    def test_render_clock_display(self, renderer):
        """Test clock at different times produces different images."""
        state1 = ScoreboardState(clock_seconds=0)
        state2 = ScoreboardState(clock_seconds=90)
        img1 = renderer.render(state1)
        img2 = renderer.render(state2)

        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert not np.array_equal(arr1, arr2)

    def test_render_top_position(self, renderer):
        """Test top position renders scoreboard near top of frame."""
        state = ScoreboardState(position="top", offset=50)
        img = renderer.render(state)
        arr = np.array(img)

        # Check that there are non-transparent pixels in top portion
        top_alpha = arr[:100, :, 3].sum()
        bottom_alpha = arr[-100:, :, 3].sum()
        assert top_alpha > bottom_alpha

    def test_cache_returns_same_image(self, renderer):
        state = ScoreboardState(home_score=1, away_score=0)
        img1 = renderer.render(state)
        img2 = renderer.render(state)
        # Should be the exact same object (cached)
        assert img1 is img2

    def test_cache_invalidated_on_change(self, renderer):
        state1 = ScoreboardState(home_score=0)
        state2 = ScoreboardState(home_score=1)
        img1 = renderer.render(state1)
        img2 = renderer.render(state2)
        assert img1 is not img2

    def test_save_to_png(self, renderer, tmp_path):
        """Render to a PNG file for visual inspection."""
        state = ScoreboardState(
            home_team="Eagles", away_team="Hawks",
            home_score=2, away_score=1,
            clock_seconds=2345,  # 39:05
            half=2,
            home_color="#1E5E3A", away_color="#5E1E1E"
        )
        img = renderer.render(state)
        png_path = str(tmp_path / "scoreboard_test.png")
        img.save(png_path)
        assert os.path.exists(png_path)
        assert os.path.getsize(png_path) > 0

    def test_composite_onto_frame(self, renderer):
        """Test compositing scoreboard onto a video frame."""
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128  # gray frame
        state = ScoreboardState(home_score=1, away_score=0)

        result = renderer.composite_onto_frame(frame, state)

        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        # Result should differ from original (scoreboard was composited)
        assert not np.array_equal(result, frame)

    def test_composite_invisible_returns_original(self, renderer):
        """Compositing invisible scoreboard should return original frame."""
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        state = ScoreboardState(visible=False)

        result = renderer.composite_onto_frame(frame, state)
        assert np.array_equal(result, frame)

    def test_composite_on_different_size_frame(self, renderer):
        """Test compositing on a non-standard frame size."""
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
        state = ScoreboardState()

        result = renderer.composite_onto_frame(frame, state)
        assert result.shape == frame.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
