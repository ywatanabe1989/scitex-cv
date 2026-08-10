#!/usr/bin/env python3
# Timestamp: 2026-08-10
# File: src/scitex_cv/_document/_types.py
"""The shape a document reading is reported in — the contract itself.

Four page states, not three
---------------------------
``text`` / ``blank`` / ``unreadable`` / ``error``. The consuming layer asked
for three (text / unreadable / error) with the explicit requirement that "a
blank page is a legitimate result and must not look like a failure, and vice
versa". Satisfying that requirement IS the fourth state: a genuinely empty
page and a page the model refused to read are different facts, and collapsing
them re-creates the ambiguity the requirement exists to remove.

``text`` is the ONLY state carrying characters. The others set ``text`` to
``None`` rather than ``""``, because an empty string is precisely the value
that cannot distinguish "nothing was written here" from "nothing was
recovered".

``unreadable`` vs ``error``
---------------------------
``unreadable`` means every rung was tried and each returned layout-only: a
fact about the PAGE. ``error`` means processing could not complete —
unreachable endpoint, corrupt page, decrypt failure: a fact about the RUN,
where the page may be perfectly readable. ``attempts`` records how many rungs
actually ran, so "fell off the end of the ladder" and "died on rung one" are
never confused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

STATUS_TEXT = "text"
STATUS_BLANK = "blank"
STATUS_UNREADABLE = "unreadable"
STATUS_ERROR = "error"
PAGE_STATUSES = (STATUS_TEXT, STATUS_BLANK, STATUS_UNREADABLE, STATUS_ERROR)

SOURCE_TEXT_LAYER = "pdf-text-layer"
SOURCE_OCR = "ocr"


@dataclass(frozen=True)
class PageReading:
    """What was observed on one page. Never a bare string.

    ``text`` is non-None if and only if ``status == "text"`` — enforced in
    ``__post_init__``, so a caller may branch on either and get the same
    answer. ``source`` says which path produced it (``pdf-text-layer`` or
    ``ocr``), which the consumer asked for explicitly.
    """

    index: int
    status: str
    text: Optional[str]
    source: Optional[str]
    rendering: Optional[Tuple[int, str]]
    image_tokens: Optional[int]
    attempts: int
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in PAGE_STATUSES:
            raise ValueError(
                f"PageReading.status must be one of {PAGE_STATUSES}, "
                f"got {self.status!r}"
            )
        if self.status == STATUS_TEXT and not self.text:
            raise ValueError(
                f"page {self.index}: status 'text' requires non-empty text; "
                "a reading with no characters is not a text reading"
            )
        if self.status != STATUS_TEXT and self.text is not None:
            raise ValueError(
                f"page {self.index}: text must be None when status is "
                f"{self.status!r} — an empty string cannot distinguish a blank "
                "page from an unrecovered one, which is the whole point"
            )
        if self.index < 0:
            raise ValueError(f"page index must be >= 0, got {self.index}")
        if self.attempts < 0:
            raise ValueError(f"attempts must be >= 0, got {self.attempts}")

    @property
    def read(self) -> bool:
        """True only when characters were recovered. A blank page is not read."""
        return self.status == STATUS_TEXT


@dataclass(frozen=True)
class DocumentReading:
    """Ordered per-page results for one input file.

    Page-level rather than document-level by request: filing rules key off
    page 1, and a 20-page register extract must not collapse into one blob.
    """

    path: str
    page_count: int
    pages: Tuple[PageReading, ...]
    ladder: Tuple[Tuple[int, str], ...]

    def __post_init__(self) -> None:
        if len(self.pages) != self.page_count:
            raise ValueError(
                f"page_count is {self.page_count} but {len(self.pages)} pages "
                "were supplied — a partial result must not claim to be whole"
            )
        for position, page in enumerate(self.pages):
            if page.index != position:
                raise ValueError(
                    "pages must be ordered and contiguous: position "
                    f"{position} holds page index {page.index}"
                )

    @property
    def unread_pages(self) -> Tuple[int, ...]:
        """Indices that produced no characters, for any reason."""
        return tuple(page.index for page in self.pages if not page.read)


# EOF
