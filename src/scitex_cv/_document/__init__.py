#!/usr/bin/env python3
# Timestamp: 2026-08-10
# File: src/scitex_cv/_document/__init__.py
"""Per-page document reading with a resumable result sidecar.

Thin orchestrator — the three responsibilities live in their own modules:

* :mod:`._types`   — the reported shape (``PageReading`` / ``DocumentReading``)
* :mod:`._sidecar` — the on-disk result and the resume policy
* :mod:`._read`    — driving a document through the render ladder

The consumer of this package decides where a document files and what it is
called. Nothing here classifies a document or generates a filename: those
depend on filing conventions, not on vision, and belong to the caller.
"""

from ._read import read_document, render_page_at_height
from ._sidecar import (
    SIDECAR_SUFFIX,
    SIDECAR_VERSION,
    ladder_fingerprint,
    read_sidecar,
    sidecar_path_for,
    write_sidecar,
)
from ._types import (
    PAGE_STATUSES,
    SOURCE_OCR,
    SOURCE_TEXT_LAYER,
    STATUS_BLANK,
    STATUS_ERROR,
    STATUS_TEXT,
    STATUS_UNREADABLE,
    DocumentReading,
    PageReading,
)

__all__ = [
    "PAGE_STATUSES",
    "SIDECAR_SUFFIX",
    "SIDECAR_VERSION",
    "SOURCE_OCR",
    "SOURCE_TEXT_LAYER",
    "STATUS_BLANK",
    "STATUS_ERROR",
    "STATUS_TEXT",
    "STATUS_UNREADABLE",
    "DocumentReading",
    "PageReading",
    "ladder_fingerprint",
    "read_document",
    "read_sidecar",
    "render_page_at_height",
    "sidecar_path_for",
    "write_sidecar",
]

# EOF
