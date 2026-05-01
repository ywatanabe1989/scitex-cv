#!/usr/bin/env python3
"""Tests for scitex_cv._filters (blur, sharpen, edge_detect, threshold, denoise)."""

import numpy as np
import pytest

from scitex_cv._filters import blur, denoise, edge_detect, sharpen, threshold


@pytest.fixture
def gray_img():
    g = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    return g


@pytest.fixture
def bgr_img():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[16:48, 16:48] = (255, 255, 255)
    return arr


class TestBlur:
    def test_gaussian_preserves_shape(self, bgr_img):
        out = blur(bgr_img, ksize=5, method="gaussian")
        assert out.shape == bgr_img.shape

    def test_even_kernel_size_is_normalized(self, bgr_img):
        out = blur(bgr_img, ksize=4, method="gaussian")
        assert out.shape == bgr_img.shape

    @pytest.mark.parametrize("method", ["gaussian", "median", "box", "bilateral"])
    def test_each_method_works(self, bgr_img, method):
        out = blur(bgr_img, ksize=5, method=method)
        assert out.shape == bgr_img.shape

    def test_unknown_method_raises(self, bgr_img):
        with pytest.raises(ValueError, match="Unknown blur"):
            blur(bgr_img, method="nonsense")


class TestSharpen:
    def test_preserves_shape(self, bgr_img):
        out = sharpen(bgr_img)
        assert out.shape == bgr_img.shape

    def test_strength_argument_accepted(self, bgr_img):
        sharpen(bgr_img, strength=1.0)
        sharpen(bgr_img, strength=2.0)


class TestEdgeDetect:
    @pytest.mark.parametrize("method", ["canny", "sobel", "laplacian"])
    def test_each_method_returns_2d(self, bgr_img, method):
        out = edge_detect(bgr_img, method=method)
        assert out.ndim == 2
        assert out.shape == bgr_img.shape[:2]

    def test_works_on_grayscale_input(self, gray_img):
        out = edge_detect(gray_img, method="canny")
        assert out.ndim == 2

    def test_unknown_method_raises(self, bgr_img):
        with pytest.raises(ValueError, match="Unknown edge"):
            edge_detect(bgr_img, method="nonsense")


class TestThreshold:
    def test_binary_produces_only_two_values(self, gray_img):
        out = threshold(gray_img, thresh=127, method="binary")
        unique = np.unique(out)
        assert set(unique) <= {0, 255}

    @pytest.mark.parametrize(
        "method",
        [
            "binary",
            "binary_inv",
            "trunc",
            "tozero",
            "tozero_inv",
            "otsu",
            "adaptive_mean",
            "adaptive_gaussian",
        ],
    )
    def test_each_method_returns_2d(self, gray_img, method):
        out = threshold(gray_img, method=method)
        assert out.ndim == 2

    def test_works_on_color_input(self, bgr_img):
        out = threshold(bgr_img, method="binary")
        assert out.ndim == 2


class TestDenoise:
    def test_fastNl_color_preserves_shape(self, bgr_img):
        out = denoise(bgr_img, strength=5, method="fastNl")
        assert out.shape == bgr_img.shape

    def test_fastNl_grayscale_preserves_shape(self, gray_img):
        out = denoise(gray_img, strength=5, method="fastNl")
        assert out.shape == gray_img.shape

    def test_bilateral_method(self, bgr_img):
        out = denoise(bgr_img, strength=5, method="bilateral")
        assert out.shape == bgr_img.shape

    def test_unknown_method_raises(self, bgr_img):
        with pytest.raises(ValueError, match="Unknown denoise"):
            denoise(bgr_img, method="nonsense")


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
