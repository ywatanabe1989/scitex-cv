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
    def test_save_returns_destination_path_unchanged(self, bgr_img, tmp_path):
        # Arrange
        path = tmp_path / "out.png"
        # Act
        result = save(bgr_img, path)
        # Assert
        assert result == path

    def test_save_creates_file_at_destination(self, bgr_img, tmp_path):
        # Arrange
        path = tmp_path / "out.png"
        # Act
        save(bgr_img, path)
        # Assert
        assert path.exists()

    def test_save_then_load_recovers_image_bitwise(self, bgr_img, tmp_path):
        # Arrange
        path = tmp_path / "out.png"
        save(bgr_img, path)
        # Act
        round_tripped = load(path)
        # Assert
        assert np.array_equal(round_tripped, bgr_img)

    def test_save_creates_missing_parent_directories(self, bgr_img, tmp_path):
        # Arrange
        path = tmp_path / "deep" / "nested" / "out.png"
        # Act
        save(bgr_img, path)
        # Assert
        assert path.exists()

    def test_load_with_color_false_returns_two_dim_array(self, bgr_img, tmp_path):
        # Arrange
        path = tmp_path / "out.png"
        save(bgr_img, path)
        # Act
        gray = load(path, color=False)
        # Assert
        assert gray.ndim == 2

    def test_load_missing_file_raises_file_not_found_error(self, tmp_path):
        # Arrange
        missing_path = tmp_path / "nope.png"
        # Act
        ctx = pytest.raises(FileNotFoundError)
        # Assert
        with ctx:
            load(missing_path)


class TestColorConversions:
    def test_to_rgb_swaps_blue_and_red_channels(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        rgb = to_rgb(source)
        # Assert
        assert tuple(rgb[0, 0]) == (50, 100, 200)

    def test_to_bgr_after_to_rgb_recovers_original_image(self, bgr_img):
        # Arrange
        rgb = to_rgb(bgr_img)
        # Act
        back = to_bgr(rgb)
        # Assert
        assert np.array_equal(back, bgr_img)

    def test_to_gray_returns_two_dimensional_array(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        gray = to_gray(source)
        # Assert
        assert gray.ndim == 2

    def test_to_gray_is_identity_on_already_grayscale_input(self):
        # Arrange
        g = np.zeros((10, 10), dtype=np.uint8)
        # Act
        out = to_gray(g)
        # Assert
        assert np.array_equal(out, g)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
