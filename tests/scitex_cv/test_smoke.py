"""Smoke tests for scitex-cv public API."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
import scitex_cv as cv


@pytest.fixture
def img_bgr():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_exports_callable():
    for name in (
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
    ):
        assert callable(getattr(cv, name)), name


def test_save_and_load_roundtrip(tmp_path, img_bgr):
    out = tmp_path / "img.png"
    cv.save(img_bgr, str(out))
    assert out.exists()
    loaded = cv.load(str(out))
    assert loaded.shape == img_bgr.shape


def test_to_gray(img_bgr):
    gray = cv.to_gray(img_bgr)
    assert gray.ndim == 2
    assert gray.shape == img_bgr.shape[:2]


def test_resize(img_bgr):
    half = cv.resize(img_bgr, scale=0.5)
    assert half.shape[0] == img_bgr.shape[0] // 2
    assert half.shape[1] == img_bgr.shape[1] // 2


def test_blur(img_bgr):
    blurred = cv.blur(img_bgr, ksize=5)
    assert blurred.shape == img_bgr.shape


def test_rectangle(img_bgr):
    drawn = cv.rectangle(img_bgr.copy(), (5, 5), (30, 30), color=(0, 255, 0))
    assert drawn.shape == img_bgr.shape
