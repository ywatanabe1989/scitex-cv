#!/usr/bin/env python3
# Timestamp: 2026-08-10
# File: src/scitex_cv/_document/_read.py
"""Drive a document through the render ladder, one page at a time.

Why each rung re-rasterizes instead of resizing one bitmap
----------------------------------------------------------
Measured 2026-08-10: for a soft scan, five renderings at an IDENTICAL image-
token count split two-to-three between reading and not, decided by the
resampling implementation and the target size interacting. Resampling is
therefore a variable that changes the answer, so this module removes one: a
PDF page is rasterized at each rung's target height DIRECTLY, rather than
rasterized once and resized per rung. A page is never resampled twice.

PDF text-layer extraction is not vision, and is here on sufferance
-------------------------------------------------------------------
If a page already carries text, no eyes are needed — that is document IO. It
lives here because no document-IO owner exists today and the consumer wants
one call rather than two. It is the first thing to move out if one appears.
Note that `scitex_cv._ocr`'s claim that "scitex-io owns PDF -> image" was
verified FALSE on 2026-08-10: scitex-io carries PDF metadata (XMP) only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .._ocr_surya import (
    SURYA_DEFAULT_ENDPOINT,
    SURYA_PROMPT,
    SURYA_RENDER_LADDER,
    is_layout_only,
)
from ._sidecar import (
    ladder_fingerprint,
    page_from_payload,
    read_sidecar,
    reusable,
    sidecar_path_for,
    write_sidecar,
)
from ._types import (
    SOURCE_OCR,
    SOURCE_TEXT_LAYER,
    STATUS_BLANK,
    STATUS_ERROR,
    STATUS_TEXT,
    STATUS_UNREADABLE,
    DocumentReading,
    PageReading,
)


def _import_pymupdf():
    """Import pymupdf lazily, with an actionable error when absent.

    Imported under its modern name rather than the ``fitz`` alias, which the
    library itself now emits a DeprecationWarning for.
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError(
            "pymupdf is required to read PDF documents but is not installed. "
            "Install it with the 'pdf' extra: pip install 'scitex-cv[pdf]'"
        ) from exc
    return pymupdf


def render_page_at_height(page, height: int):
    """Rasterize ONE pdf page so its long edge is ``height`` pixels (BGR)."""
    import numpy as np

    pymupdf = _import_pymupdf()
    rect = page.rect
    long_edge = max(rect.width, rect.height)
    if long_edge <= 0:
        raise ValueError("page has zero extent")
    zoom = height / float(long_edge)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    # pymupdf yields RGB; cv2 encodes BGR. Reverse the channel axis rather
    # than cvtColor so this stays a pure numpy operation.
    return array[:, :, ::-1].copy()


#: Long edge used for the cheap pre-OCR blankness probe. Small on purpose —
#: this only has to answer "is there ink", not "what does it say".
BLANK_PROBE_HEIGHT = 400

#: Ink coverage at or below which a page is reported ``blank``.
#:
#: MEASURED, not guessed (2026-08-10, the three real ScanSnap documents plus a
#: rendered blank page — see the card). A blank page carries scanner speckle
#: rather than pure white, so the line cannot be zero; a page with even one
#: short line of text sits orders of magnitude above this.
BLANK_INK_FRACTION = 0.002

#: A pixel darker than this counts as ink. Deliberately permissive: scanned
#: paper is grey, and a threshold near white would count the page itself.
INK_LEVEL = 200


def page_ink_fraction(image) -> float:
    """Fraction of pixels dark enough to be ink.

    Blankness is a property of the PIXELS and must not be asked of the model:
    measured 2026-08-10, Surya returns layout-only JSON for a blank page
    exactly as it does for an unreadable one, so the two are indistinguishable
    by response. Asking the image instead makes them distinguishable, is
    deterministic, and skips a full ladder walk per blank page — which matters
    for a scan stack full of separator sheets.
    """
    import cv2
    import numpy as np

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.count_nonzero(grey < INK_LEVEL)) / float(grey.size)


def _pending(index: int, detail: str) -> PageReading:
    """Placeholder for a page not yet processed, used in interim sidecars."""
    return PageReading(
        index=index,
        status=STATUS_ERROR,
        text=None,
        source=None,
        rendering=None,
        image_tokens=None,
        attempts=0,
        detail=detail,
    )


def read_document(
    path: Union[str, Path],
    endpoint: str = SURYA_DEFAULT_ENDPOINT,
    prompt: str = SURYA_PROMPT,
    ladder: Tuple[Tuple[int, str], ...] = SURYA_RENDER_LADDER,
    force: bool = False,
    max_tokens: int = 4096,
    timeout: float = 300.0,
) -> DocumentReading:
    """Read every page of ``path``, resuming from a sidecar when one exists.

    Re-running a document that already has a sidecar is a NO-OP for pages that
    were resolved, unless ``force``. That is what makes an overnight batch
    survivable.

    A PDF page carrying a real text layer is taken from that layer and OCR is
    skipped entirely — faster and more accurate — recording
    ``source="pdf-text-layer"`` so the caller can tell which path ran.

    Returns
    -------
    DocumentReading
        Always. A page that cannot be processed becomes an ``error`` page
        rather than an exception: one bad page in a 300-page scan must not
        discard the other 299. Only a document that cannot be OPENED raises.
    """
    document_path = Path(path)
    sidecar = sidecar_path_for(document_path)
    ladder = tuple(ladder)
    fingerprint = ladder_fingerprint(ladder)

    previous: Dict[int, PageReading] = {}
    if not force:
        payload = read_sidecar(sidecar)
        if payload is not None:
            stored = payload.get("ladder_fingerprint", "")
            for entry in payload.get("pages", []):
                if not isinstance(entry, dict):
                    continue
                page = page_from_payload(entry)
                if page is not None and reusable(page, stored, fingerprint):
                    previous[page.index] = page

    pymupdf = _import_pymupdf()
    try:
        document = pymupdf.open(str(document_path))
    except Exception as exc:
        raise OSError(
            f"could not open {document_path} as a document ({exc}). If it is a "
            "plain image rather than a PDF, call ocr_surya() on it directly."
        ) from exc

    pages: List[PageReading] = []
    try:
        total = document.page_count
        for index in range(total):
            carried = previous.get(index)
            pages.append(
                carried
                if carried is not None
                else _read_one_page(
                    document,
                    index,
                    ladder=ladder,
                    endpoint=endpoint,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            )
            # Persist after EVERY page, not once at the end. A sidecar written
            # only on completion is useless in exactly the case it exists for.
            write_sidecar(
                sidecar,
                DocumentReading(
                    path=str(document_path),
                    page_count=total,
                    pages=tuple(pages)
                    + tuple(
                        _pending(i, "not yet processed")
                        for i in range(len(pages), total)
                    ),
                    ladder=ladder,
                ),
            )
        reading = DocumentReading(
            path=str(document_path),
            page_count=total,
            pages=tuple(pages),
            ladder=ladder,
        )
    finally:
        document.close()

    write_sidecar(sidecar, reading)
    return reading


def _read_one_page(
    document,
    index: int,
    *,
    ladder: Tuple[Tuple[int, str], ...],
    endpoint: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> PageReading:
    """Resolve one page: text layer if present, else walk the ladder."""
    from .._ocr_surya import _ask, _classify

    try:
        page = document[index]
    except Exception as exc:
        return PageReading(
            index=index,
            status=STATUS_ERROR,
            text=None,
            source=None,
            rendering=None,
            image_tokens=None,
            attempts=0,
            detail=f"page could not be loaded: {exc}",
        )

    try:
        embedded = page.get_text().strip()
    except Exception:
        embedded = ""
    if embedded:
        return PageReading(
            index=index,
            status=STATUS_TEXT,
            text=embedded,
            source=SOURCE_TEXT_LAYER,
            rendering=None,
            image_tokens=None,
            attempts=0,
        )

    # Blankness is settled from the pixels BEFORE the ladder runs. Asking the
    # model cannot answer it — a blank page and an unreadable one produce the
    # same layout-only response — and asking it anyway costs a full ladder
    # walk per blank page.
    try:
        probe = render_page_at_height(page, BLANK_PROBE_HEIGHT)
        ink = page_ink_fraction(probe)
    except Exception as exc:
        return PageReading(
            index=index,
            status=STATUS_ERROR,
            text=None,
            source=None,
            rendering=None,
            image_tokens=None,
            attempts=0,
            detail=f"blankness probe failed: {exc}",
        )
    if ink <= BLANK_INK_FRACTION:
        return PageReading(
            index=index,
            status=STATUS_BLANK,
            text=None,
            source=SOURCE_OCR,
            rendering=None,
            image_tokens=None,
            attempts=0,
            detail=f"ink coverage {ink:.5f} <= {BLANK_INK_FRACTION}",
        )

    last_tokens: Optional[int] = None
    attempts = 0
    for rung in ladder:
        height, _filter_name = rung
        try:
            image = render_page_at_height(page, height)
        except Exception as exc:
            return PageReading(
                index=index,
                status=STATUS_ERROR,
                text=None,
                source=None,
                rendering=rung,
                image_tokens=None,
                attempts=attempts,
                detail=f"rasterization failed on rung {attempts}: {exc}",
            )
        try:
            body, tokens = _ask(image, endpoint, prompt, max_tokens, timeout)
        except ConnectionError as exc:
            # A transport failure says nothing about the page. Reporting it as
            # `unreadable` would blame the document for a server outage.
            return PageReading(
                index=index,
                status=STATUS_ERROR,
                text=None,
                source=None,
                rendering=rung,
                image_tokens=None,
                attempts=attempts,
                detail=f"endpoint unreachable on rung {attempts}: {exc}",
            )
        attempts += 1
        last_tokens = tokens
        if _classify(body) == "text":
            return PageReading(
                index=index,
                status=STATUS_TEXT,
                text=body,
                source=SOURCE_OCR,
                rendering=rung,
                image_tokens=tokens,
                attempts=attempts,
            )
        if not body.strip() and not is_layout_only(body):
            # Nothing at all came back, rather than block geometry. Blankness
            # is normally settled by the ink probe above, so reaching here
            # means the model returned an empty body for a page that DOES
            # carry ink — reported as blank because that is what was observed,
            # not because it was expected.
            return PageReading(
                index=index,
                status=STATUS_BLANK,
                text=None,
                source=SOURCE_OCR,
                rendering=rung,
                image_tokens=tokens,
                attempts=attempts,
                detail="model returned an empty body despite ink on the page",
            )

    return PageReading(
        index=index,
        status=STATUS_UNREADABLE,
        text=None,
        source=SOURCE_OCR,
        rendering=ladder[-1] if ladder else None,
        image_tokens=last_tokens,
        attempts=attempts,
        detail=f"all {attempts} rung(s) returned layout-only",
    )


# EOF
