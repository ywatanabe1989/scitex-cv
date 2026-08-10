#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/cv/__init__.py
"""scitex-cv — small cv2-based image utilities (standalone).

Provides reusable cv2-based utilities for image processing:
- I/O: load, save, color conversions
- Transform: resize, rotate, flip, crop, pad
- Filters: blur, sharpen, edge detection, threshold, denoise
- Draw: rectangle, circle, line, text, polylines, arrow
- OCR: ocr (image -> text via EasyOCR; optional `ocr` extra)
- OCR: ocr_surya (image -> structured layout+text via a Surya-2 llama-server;
  no torch, so it runs on GPUs current torch wheels ship no kernels for)

Example
-------
>>> import scitex.cv as cv
>>> # Load and process an image
>>> img = cv.load("input.png")
>>> img = cv.resize(img, scale=0.5)
>>> img = cv.blur(img, ksize=5)
>>> edges = cv.edge_detect(img, method="canny")
>>> cv.save(edges, "edges.png")
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v

try:
    __version__ = _v("scitex-cv")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

del _v, PackageNotFoundError

# Public name -> submodule that defines it. Imports are deferred (PEP 562)
# so that `import scitex_cv` does NOT pull in cv2 (and its OS shared libs
# libxcb/libgl/libglib) at package-load time. This keeps the package — and
# the `scitex_dev.system_deps` provider in ._system_deps — importable in a
# minimal/build environment that does not yet have those libs, so the
# ecosystem aggregator can discover them via the entry point. Accessing a
# function (e.g. ``scitex_cv.load``) imports cv2 on first use, as before.
_SUBMODULE_BY_NAME = {
    "arrow": "._draw",
    "circle": "._draw",
    "line": "._draw",
    "polylines": "._draw",
    "rectangle": "._draw",
    "text": "._draw",
    "blur": "._filters",
    "denoise": "._filters",
    "edge_detect": "._filters",
    "sharpen": "._filters",
    "threshold": "._filters",
    "load": "._io",
    "save": "._io",
    "to_bgr": "._io",
    "to_gray": "._io",
    "to_rgb": "._io",
    "crop": "._transform",
    "flip": "._transform",
    "pad": "._transform",
    "resize": "._transform",
    "rotate": "._transform",
    "ocr": "._ocr",
    "torch_build_has_kernels_for_local_gpu": "._ocr",
    "SuryaReading": "._ocr_surya",
    "is_layout_only": "._ocr_surya",
    "normalize_page": "._ocr_surya",
    "ocr_surya": "._ocr_surya",
    "DocumentReading": "._document",
    "PageReading": "._document",
    "read_document": "._document",
    "sidecar_path_for": "._document",
}


def __getattr__(name: str):
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr = getattr(importlib.import_module(submodule, __name__), name)
    globals()[name] = attr  # cache so later lookups skip __getattr__
    return attr


def __dir__():
    return sorted(__all__)

__all__ = [
    "__version__",
    # I/O
    "load",
    "save",
    "to_rgb",
    "to_bgr",
    "to_gray",
    # Transforms
    "resize",
    "rotate",
    "flip",
    "crop",
    "pad",
    # Filters
    "blur",
    "sharpen",
    "edge_detect",
    "threshold",
    "denoise",
    # Drawing
    "rectangle",
    "circle",
    "line",
    "text",
    "polylines",
    "arrow",
    # OCR
    "ocr",
    "torch_build_has_kernels_for_local_gpu",
    # OCR — Surya-2 engine (llama-server; no torch)
    "ocr_surya",
    "SuryaReading",
    "normalize_page",
    "is_layout_only",
    # Documents — PDF in, ordered per-page results out, resumable
    "read_document",
    "DocumentReading",
    "PageReading",
    "sidecar_path_for",
]

# EOF
