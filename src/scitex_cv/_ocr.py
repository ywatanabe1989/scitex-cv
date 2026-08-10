#!/usr/bin/env python3
# Timestamp: 2026-07-18
# File: src/scitex_cv/_ocr.py
"""Optical character recognition (image -> text) using EasyOCR.

OCR is a natural CV primitive: it maps an image (a file path or a cv2/numpy
array) to recognized text. The heavy engine (EasyOCR, which pulls in torch)
is imported **lazily inside** :func:`ocr` so that ``import scitex_cv`` stays
light and free of the optional dependency.

PDF handling deliberately lives elsewhere (scitex-io owns PDF -> image);
this module only ever sees images, keeping separation of concerns clean.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

_OCR_EXTRA_HINT = (
    "EasyOCR is required for scitex_cv.ocr but is not installed. "
    "Install it with the 'ocr' extra: pip install 'scitex-cv[ocr]'"
)


def torch_build_has_kernels_for_local_gpu() -> bool:
    """Whether the INSTALLED TORCH BUILD ships kernels for THIS machine's GPU.

    Deliberately named after torch rather than after "GPU availability",
    because the two are not the same question and conflating them is the bug
    this exists to prevent.

    ``torch.cuda.is_available()`` answers "is there a working driver and
    runtime", and on a GTX 1070 it returns **True** — while
    ``torch.cuda.get_arch_list()`` for torch 2.13.0+cu130 is
    ``sm_75/80/86/90/100/120`` and the card is ``sm_61``. The driver is fine;
    only the kernels for this card are missing. So the obvious probe reports a
    usable GPU and every torch-backed engine then selects one that cannot run.
    (Measured by the grant agent on scitex-compute-03, 2026-08-10.)

    The name carries the scope boundary on purpose: this is a fact about a
    torch build, NOT about the card. Engines on other runtimes — CTranslate2,
    ONNX Runtime, llama.cpp — have their own kernel coverage and must not
    inherit this answer. In particular llama.cpp built with
    ``CMAKE_CUDA_ARCHITECTURES=61`` runs perfectly well on the same card.

    Returns False when torch is absent, since no torch build then has kernels
    for anything.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


@lru_cache(maxsize=None)
def _get_reader(languages: Tuple[str, ...], gpu: bool):
    """Build and cache an EasyOCR ``Reader`` for a language set.

    The reader is cached per (language tuple, gpu) because model load is slow.
    EasyOCR is imported here (never at module top) so that torch is only
    pulled in when OCR is actually requested.
    """
    easyocr = _import_easyocr()
    return easyocr.Reader(list(languages), gpu=gpu)


def _import_easyocr():
    """Import easyocr lazily, raising an actionable error when absent."""
    try:
        import easyocr
    except ImportError as exc:  # dependency genuinely absent
        raise ImportError(_OCR_EXTRA_HINT) from exc
    return easyocr


def ocr(
    image: Union[str, Path, np.ndarray],
    languages: Sequence[str] = ("ja", "en"),
    detail: bool = False,
    gpu: Optional[bool] = None,
) -> Union[str, List[Tuple]]:
    """Recognize text in an image.

    Parameters
    ----------
    image : str, Path, or np.ndarray
        A path to an image file, or an in-memory image array. Arrays may be
        BGR (cv2's native order, as produced by :func:`scitex_cv.load`) or
        RGB / grayscale — EasyOCR handles any of these numpy inputs directly.
    languages : sequence of str
        Language codes to recognize (EasyOCR codes, e.g. ``"en"``, ``"ja"``).
        Defaults to Japanese + English. The set is used as the cache key for
        the underlying model, so the same tuple reuses one loaded ``Reader``.
    detail : bool
        If False (default), return the recognized text pieces concatenated
        into a single string. If True, return the raw list of
        ``(bbox, text, confidence)`` tuples EasyOCR yields.
    gpu : bool, optional
        Whether EasyOCR should use CUDA. Left as None (the default) it is
        decided by :func:`torch_build_has_kernels_for_local_gpu`, NOT by
        EasyOCR's own default of ``True``. EasyOCR trusts
        ``torch.cuda.is_available()``, which answers True on cards this torch
        build ships no kernels for (a GTX 1070 being the measured case), so
        the default selects a GPU that cannot run the model. Pass True or
        False to override the probe.

    Returns
    -------
    str or list
        Concatenated recognized text (``detail=False``) or the list of
        ``(bbox, text, confidence)`` tuples (``detail=True``).

    Raises
    ------
    TypeError
        If ``image`` is neither a path-like nor a numpy array.
    ImportError
        If EasyOCR is not installed (install ``scitex-cv[ocr]``).
    """
    if isinstance(image, np.ndarray):
        target: Union[str, np.ndarray] = image
    elif isinstance(image, (str, Path)):
        target = str(image)
    else:
        raise TypeError(
            "ocr(image): image must be a file path (str/Path) or a numpy "
            f"array, got {type(image).__name__}"
        )

    if gpu is None:
        gpu = torch_build_has_kernels_for_local_gpu()

    reader = _get_reader(tuple(languages), gpu)
    results = reader.readtext(target)

    if detail:
        return results
    return " ".join(text for _bbox, text, _conf in results)


__all__ = ["ocr", "torch_build_has_kernels_for_local_gpu"]

# EOF
