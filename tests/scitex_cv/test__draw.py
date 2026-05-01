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
    def test_outlined_draws_pixels(self, canvas):
        out = rectangle(canvas, (10, 10), (50, 50), color=(0, 255, 0))
        assert _has_nonzero(out)
        assert out[10, 10, 1] == 255

    def test_filled_fills_interior(self, canvas):
        out = rectangle(canvas, (10, 10), (50, 50), color=(0, 0, 255), filled=True)
        assert tuple(out[30, 30]) == (0, 0, 255)

    def test_returns_same_array(self, canvas):
        out = rectangle(canvas, (0, 0), (10, 10))
        assert out is canvas


class TestCircle:
    def test_outlined_draws_perimeter(self, canvas):
        out = circle(canvas, (50, 50), radius=20, color=(255, 0, 0))
        assert _has_nonzero(out)

    def test_filled_fills_disc(self, canvas):
        out = circle(canvas, (50, 50), radius=20, filled=True, color=(0, 0, 255))
        assert tuple(out[50, 50]) == (0, 0, 255)


class TestLine:
    def test_diagonal_line(self, canvas):
        out = line(canvas, (0, 0), (99, 99), color=(0, 255, 0), thickness=1)
        diag = np.array([out[i, i] for i in range(100)])
        assert diag.any()


class TestText:
    def test_writes_some_pixels(self, canvas):
        out = text(canvas, "Hi", (10, 50), color=(255, 255, 255))
        assert _has_nonzero(out)


class TestPolylines:
    def test_open_polyline(self, canvas):
        pts = np.array([[10, 10], [50, 10], [50, 50]], dtype=np.int32)
        out = polylines(canvas, pts, color=(0, 255, 0), closed=False)
        assert _has_nonzero(out)

    def test_closed_polygon(self, canvas):
        pts = np.array([[10, 10], [50, 10], [50, 50]], dtype=np.int32)
        out = polylines(canvas, pts, color=(0, 255, 0), closed=True)
        assert _has_nonzero(out)


class TestArrow:
    def test_arrow_draws_pixels(self, canvas):
        out = arrow(canvas, (10, 10), (80, 80), color=(255, 0, 0))
        assert _has_nonzero(out)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
