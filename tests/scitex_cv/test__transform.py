#!/usr/bin/env python3
"""Tests for scitex_cv._transform (resize, rotate, flip, crop, pad)."""

import numpy as np
import pytest

from scitex_cv._transform import crop, flip, pad, resize, rotate


@pytest.fixture
def img():
    arr = np.zeros((60, 100, 3), dtype=np.uint8)
    arr[10:20, 10:20] = (255, 0, 0)
    return arr


class TestResize:
    def test_resize_to_explicit_size_produces_requested_shape(self, img):
        # Arrange
        source = img
        # Act
        out = resize(source, size=(50, 30))
        # Assert
        assert out.shape == (30, 50, 3)

    def test_resize_by_scale_half_halves_each_axis(self, img):
        # Arrange
        source = img
        # Act
        out = resize(source, scale=0.5)
        # Assert
        assert out.shape == (30, 50, 3)

    def test_resize_without_size_or_scale_raises_value_error(self, img):
        # Arrange
        source = img
        # Act
        ctx = pytest.raises(ValueError, match="size or scale")
        # Assert
        with ctx:
            resize(source)


class TestRotate:
    def test_180_degree_rotation_preserves_image_shape(self, img):
        # Arrange
        source = img
        # Act
        out = rotate(source, angle=180)
        # Assert
        assert out.shape == source.shape

    def test_zero_degree_rotation_returns_equivalent_image(self, img):
        # Arrange
        source = img
        # Act
        out = rotate(source, angle=0)
        # Assert
        assert np.array_equal(out, source)


class TestFlip:
    def test_horizontal_flip_reverses_columns(self, img):
        # Arrange
        source = img
        # Act
        out = flip(source, "horizontal")
        # Assert
        assert np.array_equal(out, source[:, ::-1])

    def test_vertical_flip_reverses_rows(self, img):
        # Arrange
        source = img
        # Act
        out = flip(source, "vertical")
        # Assert
        assert np.array_equal(out, source[::-1])

    def test_both_flip_reverses_rows_and_columns(self, img):
        # Arrange
        source = img
        # Act
        out = flip(source, "both")
        # Assert
        assert np.array_equal(out, source[::-1, ::-1])


class TestCrop:
    def test_crop_returns_region_with_requested_shape(self, img):
        # Arrange
        source = img
        # Act
        out = crop(source, x=10, y=10, width=10, height=10)
        # Assert
        assert out.shape == (10, 10, 3)

    def test_crop_returns_pixels_from_correct_region(self, img):
        # Arrange
        source = img
        # Act
        out = crop(source, x=10, y=10, width=10, height=10)
        # Assert
        assert (out == (255, 0, 0)).all()

    def test_crop_returns_independent_copy_not_view(self, img):
        # Arrange
        source = img
        out = crop(source, x=0, y=0, width=5, height=5)
        # Act
        out[0, 0] = (1, 2, 3)
        # Assert
        assert tuple(source[0, 0]) == (0, 0, 0)


class TestPad:
    def test_pad_increases_output_dimensions_by_pad_amounts(self, img):
        # Arrange
        source = img
        # Act
        out = pad(source, top=5, bottom=5, left=10, right=10)
        # Assert
        assert out.shape == (60 + 10, 100 + 20, 3)

    def test_pad_default_color_zero_fills_padding_with_zero(self, img):
        # Arrange
        source = img
        # Act
        out = pad(source, top=2, color=0)
        # Assert
        assert (out[0:2] == 0).all()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
