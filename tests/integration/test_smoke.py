"""Smoke tests for scitex-cv public API."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
import scitex_cv as cv  # noqa: E402

_EXPORTED_NAMES = (
    "load",
    "save",
    "to_bgr",
    "to_gray",
    "to_rgb",
    "resize",
    "rotate",
    "flip",
    "crop",
    "pad",
    "blur",
    "sharpen",
    "edge_detect",
    "threshold",
    "denoise",
    "rectangle",
    "circle",
    "line",
    "polylines",
    "arrow",
    "text",
)


@pytest.fixture
def img_bgr():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", _EXPORTED_NAMES)
def test_public_export_is_callable(name):
    # Arrange
    module = cv
    # Act
    attr = getattr(module, name)
    # Assert
    assert callable(attr), name


def test_save_writes_image_file_to_disk(tmp_path, img_bgr):
    # Arrange
    out = tmp_path / "img.png"
    # Act
    cv.save(img_bgr, str(out))
    # Assert
    assert out.exists()


def test_load_returns_image_with_same_shape_as_saved(tmp_path, img_bgr):
    # Arrange
    out = tmp_path / "img.png"
    cv.save(img_bgr, str(out))
    # Act
    loaded = cv.load(str(out))
    # Assert
    assert loaded.shape == img_bgr.shape


def test_to_gray_returns_two_dimensional_array(img_bgr):
    # Arrange
    source = img_bgr
    # Act
    gray = cv.to_gray(source)
    # Assert
    assert gray.ndim == 2


def test_to_gray_preserves_height_and_width(img_bgr):
    # Arrange
    source = img_bgr
    # Act
    gray = cv.to_gray(source)
    # Assert
    assert gray.shape == source.shape[:2]


def test_resize_halves_height_when_scale_is_half(img_bgr):
    # Arrange
    source = img_bgr
    # Act
    half = cv.resize(source, scale=0.5)
    # Assert
    assert half.shape[0] == source.shape[0] // 2


def test_resize_halves_width_when_scale_is_half(img_bgr):
    # Arrange
    source = img_bgr
    # Act
    half = cv.resize(source, scale=0.5)
    # Assert
    assert half.shape[1] == source.shape[1] // 2


def test_blur_preserves_input_image_shape(img_bgr):
    # Arrange
    source = img_bgr
    # Act
    blurred = cv.blur(source, ksize=5)
    # Assert
    assert blurred.shape == source.shape


def test_rectangle_preserves_input_image_shape(img_bgr):
    # Arrange
    source = img_bgr.copy()
    # Act
    drawn = cv.rectangle(source, (5, 5), (30, 30), color=(0, 255, 0))
    # Assert
    assert drawn.shape == img_bgr.shape
