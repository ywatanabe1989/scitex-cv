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
    def test_resize_to_explicit_size(self, img):
        out = resize(img, size=(50, 30))
        assert out.shape == (30, 50, 3)

    def test_resize_by_scale(self, img):
        out = resize(img, scale=0.5)
        assert out.shape == (30, 50, 3)

    def test_requires_size_or_scale(self, img):
        with pytest.raises(ValueError, match="size or scale"):
            resize(img)


class TestRotate:
    def test_180_rotation_preserves_shape(self, img):
        out = rotate(img, angle=180)
        assert out.shape == img.shape

    def test_zero_rotation_returns_equivalent_image(self, img):
        out = rotate(img, angle=0)
        np.testing.assert_array_equal(out, img)


class TestFlip:
    def test_horizontal_flip(self, img):
        out = flip(img, "horizontal")
        np.testing.assert_array_equal(out, img[:, ::-1])

    def test_vertical_flip(self, img):
        out = flip(img, "vertical")
        np.testing.assert_array_equal(out, img[::-1])

    def test_both_flip(self, img):
        out = flip(img, "both")
        np.testing.assert_array_equal(out, img[::-1, ::-1])


class TestCrop:
    def test_crop_returns_correct_region(self, img):
        out = crop(img, x=10, y=10, width=10, height=10)
        assert out.shape == (10, 10, 3)
        assert (out == (255, 0, 0)).all()

    def test_crop_returns_copy(self, img):
        out = crop(img, x=0, y=0, width=5, height=5)
        out[0, 0] = (1, 2, 3)
        assert tuple(img[0, 0]) == (0, 0, 0)


class TestPad:
    def test_pad_increases_dimensions(self, img):
        out = pad(img, top=5, bottom=5, left=10, right=10)
        assert out.shape == (60 + 10, 100 + 20, 3)

    def test_pad_default_color_is_zero(self, img):
        out = pad(img, top=2, color=0)
        assert (out[0:2] == 0).all()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
