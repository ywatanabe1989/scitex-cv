#!/usr/bin/env python3
"""Tests for scitex_cv._io (load, save, to_rgb, to_bgr, to_gray)."""

import numpy as np
import pytest

from scitex_cv._io import load, save, to_bgr, to_gray, to_rgb


@pytest.fixture
def bgr_img():
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    arr[..., 0] = 200
    arr[..., 1] = 100
    arr[..., 2] = 50
    return arr


class TestRoundTrip:
    def test_save_then_load_recovers_image(self, bgr_img, tmp_path):
        path = tmp_path / "out.png"
        result = save(bgr_img, path)
        assert result == path
        assert path.exists()
        round_tripped = load(path)
        np.testing.assert_array_equal(round_tripped, bgr_img)

    def test_save_creates_parent_dirs(self, bgr_img, tmp_path):
        path = tmp_path / "deep" / "nested" / "out.png"
        save(bgr_img, path)
        assert path.exists()

    def test_load_grayscale(self, bgr_img, tmp_path):
        path = tmp_path / "out.png"
        save(bgr_img, path)
        gray = load(path, color=False)
        assert gray.ndim == 2

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nope.png")


class TestColorConversions:
    def test_to_rgb_swaps_b_and_r(self, bgr_img):
        rgb = to_rgb(bgr_img)
        assert tuple(rgb[0, 0]) == (50, 100, 200)

    def test_to_bgr_round_trip(self, bgr_img):
        rgb = to_rgb(bgr_img)
        back = to_bgr(rgb)
        np.testing.assert_array_equal(back, bgr_img)

    def test_to_gray_returns_2d(self, bgr_img):
        gray = to_gray(bgr_img)
        assert gray.ndim == 2

    def test_to_gray_idempotent_on_grayscale(self):
        g = np.zeros((10, 10), dtype=np.uint8)
        out = to_gray(g)
        np.testing.assert_array_equal(out, g)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
