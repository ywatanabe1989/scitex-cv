#!/usr/bin/env python3
"""Tests for scitex_cv._draw primitives (rectangle, circle, line, text, polylines, arrow)."""

import numpy as np
import pytest

from scitex_cv._draw import arrow, circle, line, polylines, rectangle, text


@pytest.fixture
def canvas():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _has_nonzero(img):
    return img.any()


class TestRectangle:
    def test_outlined_rectangle_draws_nonzero_pixels(self, canvas):
        # Arrange
        target = canvas
        # Act
        out = rectangle(target, (10, 10), (50, 50), color=(0, 255, 0))
        # Assert
        assert _has_nonzero(out)

    def test_outlined_rectangle_paints_top_left_corner_with_color(self, canvas):
        # Arrange
        target = canvas
        # Act
        out = rectangle(target, (10, 10), (50, 50), color=(0, 255, 0))
        # Assert
        assert out[10, 10, 1] == 255

    def test_filled_rectangle_fills_interior_pixels_with_color(self, canvas):
        # Arrange
        target = canvas
        # Act
        out = rectangle(target, (10, 10), (50, 50), color=(0, 0, 255), filled=True)
        # Assert
        assert tuple(out[30, 30]) == (0, 0, 255)

    def test_rectangle_returns_same_array_object_inplace(self, canvas):
        # Arrange
        target = canvas
        # Act
        out = rectangle(target, (0, 0), (10, 10))
        # Assert
        assert out is target


class TestCircle:
    def test_outlined_circle_draws_perimeter_pixels(self, canvas):
        # Arrange
        target = canvas
        # Act
        out = circle(target, (50, 50), radius=20, color=(255, 0, 0))
        # Assert
        assert _has_nonzero(out)

    def test_filled_circle_fills_center_with_color(self, canvas):
        # Arrange
        target = canvas
        # Act
        out = circle(target, (50, 50), radius=20, filled=True, color=(0, 0, 255))
        # Assert
        assert tuple(out[50, 50]) == (0, 0, 255)


class TestLine:
    def test_diagonal_line_paints_pixels_along_diagonal(self, canvas):
        # Arrange
        target = canvas
        # Act
        out = line(target, (0, 0), (99, 99), color=(0, 255, 0), thickness=1)
        # Assert
        diag = np.array([out[i, i] for i in range(100)])
        assert diag.any()


class TestText:
    def test_text_renders_at_least_some_pixels(self, canvas):
        # Arrange
        target = canvas
        # Act
        out = text(target, "Hi", (10, 50), color=(255, 255, 255))
        # Assert
        assert _has_nonzero(out)


class TestPolylines:
    def test_open_polyline_draws_nonzero_pixels(self, canvas):
        # Arrange
        pts = np.array([[10, 10], [50, 10], [50, 50]], dtype=np.int32)
        # Act
        out = polylines(canvas, pts, color=(0, 255, 0), closed=False)
        # Assert
        assert _has_nonzero(out)

    def test_closed_polygon_draws_nonzero_pixels(self, canvas):
        # Arrange
        pts = np.array([[10, 10], [50, 10], [50, 50]], dtype=np.int32)
        # Act
        out = polylines(canvas, pts, color=(0, 255, 0), closed=True)
        # Assert
        assert _has_nonzero(out)


class TestArrow:
    def test_arrow_draws_nonzero_pixels_on_canvas(self, canvas):
        # Arrange
        target = canvas
        # Act
        out = arrow(target, (10, 10), (80, 80), color=(255, 0, 0))
        # Assert
        assert _has_nonzero(out)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
