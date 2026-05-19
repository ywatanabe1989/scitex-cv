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
    def test_gaussian_blur_preserves_image_shape(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        out = blur(source, ksize=5, method="gaussian")
        # Assert
        assert out.shape == source.shape

    def test_even_kernel_size_is_normalized_internally(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        out = blur(source, ksize=4, method="gaussian")
        # Assert
        assert out.shape == source.shape

    @pytest.mark.parametrize("method", ["gaussian", "median", "box", "bilateral"])
    def test_each_blur_method_preserves_shape(self, bgr_img, method):
        # Arrange
        source = bgr_img
        # Act
        out = blur(source, ksize=5, method=method)
        # Assert
        assert out.shape == source.shape

    def test_unknown_blur_method_raises_value_error(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        ctx = pytest.raises(ValueError, match="Unknown blur")
        # Assert
        with ctx:
            blur(source, method="nonsense")


class TestSharpen:
    def test_sharpen_preserves_image_shape(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        out = sharpen(source)
        # Assert
        assert out.shape == source.shape

    @pytest.mark.parametrize("strength", [1.0, 2.0])
    def test_sharpen_accepts_strength_argument_without_error(self, bgr_img, strength):
        # Arrange
        source = bgr_img
        # Act
        out = sharpen(source, strength=strength)
        # Assert
        assert out is not None


class TestEdgeDetect:
    @pytest.mark.parametrize("method", ["canny", "sobel", "laplacian"])
    def test_each_edge_method_returns_two_dimensional_array(self, bgr_img, method):
        # Arrange
        source = bgr_img
        # Act
        out = edge_detect(source, method=method)
        # Assert
        assert out.ndim == 2

    @pytest.mark.parametrize("method", ["canny", "sobel", "laplacian"])
    def test_each_edge_method_preserves_height_and_width(self, bgr_img, method):
        # Arrange
        source = bgr_img
        # Act
        out = edge_detect(source, method=method)
        # Assert
        assert out.shape == source.shape[:2]

    def test_edge_detect_canny_works_on_grayscale_input(self, gray_img):
        # Arrange
        source = gray_img
        # Act
        out = edge_detect(source, method="canny")
        # Assert
        assert out.ndim == 2

    def test_unknown_edge_method_raises_value_error(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        ctx = pytest.raises(ValueError, match="Unknown edge")
        # Assert
        with ctx:
            edge_detect(source, method="nonsense")


class TestThreshold:
    def test_binary_threshold_produces_only_zero_and_255_values(self, gray_img):
        # Arrange
        source = gray_img
        # Act
        out = threshold(source, thresh=127, method="binary")
        # Assert
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
    def test_each_threshold_method_returns_two_dimensional_array(
        self, gray_img, method
    ):
        # Arrange
        source = gray_img
        # Act
        out = threshold(source, method=method)
        # Assert
        assert out.ndim == 2

    def test_threshold_works_on_color_input_returns_two_dim(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        out = threshold(source, method="binary")
        # Assert
        assert out.ndim == 2


class TestDenoise:
    def test_fastNl_denoise_on_color_preserves_shape(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        out = denoise(source, strength=5, method="fastNl")
        # Assert
        assert out.shape == source.shape

    def test_fastNl_denoise_on_grayscale_preserves_shape(self, gray_img):
        # Arrange
        source = gray_img
        # Act
        out = denoise(source, strength=5, method="fastNl")
        # Assert
        assert out.shape == source.shape

    def test_bilateral_denoise_preserves_shape(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        out = denoise(source, strength=5, method="bilateral")
        # Assert
        assert out.shape == source.shape

    def test_unknown_denoise_method_raises_value_error(self, bgr_img):
        # Arrange
        source = bgr_img
        # Act
        ctx = pytest.raises(ValueError, match="Unknown denoise")
        # Assert
        with ctx:
            denoise(source, method="nonsense")


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
