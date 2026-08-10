#!/usr/bin/env python3
# Timestamp: 2026-08-10
# File: src/scitex_cv/_ocr_surya.py
"""Surya-2 OCR engine, served by llama.cpp's ``llama-server``.

A second engine alongside :mod:`scitex_cv._ocr` (EasyOCR). It talks HTTP to a
running ``llama-server`` loaded with the Surya-2 GGUF plus its multimodal
projector, so it pulls in **no torch** — which is why it works on Pascal cards
(GTX 1070, sm_61) that current torch wheels ship no kernels for.

Two behaviours here are measured, not assumed (2026-08-10, GTX 1070,
surya-ocr-2-gguf, three real scanned documents):

**1. The model reads only near one working resolution.**
Off it, the model does not error — it silently answers a *different question*,
returning layout analysis with no characters. Measured, image tokens in
parentheses:

===================  ==============  ==============
page                 sharp source    soft scan
===================  ==============  ==============
753x1073   (~900)    layout-only     layout-only
1241x1755  (2201)    text            text
1506x2146  (3205)    text            **layout-only**
1861x2632  (4084)    text            (not measured)
===================  ==============  ==============

So the lower bound is universal, but the upper bound depends on how sharp the
source is: at the *same* 3205 tokens a crisp page reads and a soft scan does
not.

**Token count is NOT the control variable, and resolution alone is not a
rule.** An earlier draft of this module claimed ``SURYA_PAGE_HEIGHT`` was an
operating point that worked for every input. That claim was FALSIFIED by
end-to-end measurement: five renderings of the same soft scan, all landing on
exactly 2201 image tokens, split two-to-three between reading and not.

=========================  ========  ========
rendering (same page)      tokens    result
=========================  ========  ========
PIL LANCZOS 1242x1770      2201      text
PIL LANCZOS 1232x1755      2201      layout
cv2 LANCZOS4 1242x1770     2201      layout
cv2 LANCZOS4 1232x1755     2201      layout
cv2 CUBIC 1242x1770        2201      text
=========================  ========  ========

Neither the resampling filter nor the target size predicts the outcome on its
own — they interact, and the model sits near a decision boundary between its
OCR and layout modes for this page. Do not read :data:`SURYA_PAGE_HEIGHT` as a
threshold that guarantees anything; it is a starting rendering, and
:func:`normalize_page` is one way of producing a candidate, not a fix.

What IS reliable is :func:`is_layout_only`: the failure is mechanically
detectable from the response, which is what lets a caller respond to it
instead of trusting a single rendering.

**2. A page whose text was not recovered is still recoverable.**
The soft scan that failed at its native 753x1073 read correctly once resampled
to 1242x1770 — the information was in the file all along, merely presented off
the model's working resolution. Do not conclude from a failure that the scan
was too coarse to contain the text.

Consequently this module never reports *why* a page did not read: overshooting
and undershooting produce the identical symptom, so any cause would be a guess.
It reports what was observed and the size actually sent, and leaves inference
to the caller.

**The design that follows: retry, do not theorise.** Twelve renderings (four
filters x three target heights) were measured against both pages:

* crisp page — **12 of 12** produced text. Sharp input is robust, and no
  preprocessing anxiety is warranted for it.
* soft scan — **2 of 12** produced text.

Two of twelve is a low hit rate, but it is not zero, and that is what matters:
a page that fails ONE rendering is not an unreadable page. So
:func:`ocr_surya` walks :data:`SURYA_RENDER_LADDER`, stops at the first
rendering whose response is not layout-only, and reports which rung won and how
many were tried. That exploits a measured redundancy instead of claiming to
understand the boundary.

The ladder's ORDER is empirical and its evidence is THIN — it is ordered by
what won on ONE soft page, and should be re-derived when more soft pages
exist. The *structure* (try several, stop at the first success, report failure
only after exhausting them) does not depend on that ordering being optimal.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

#: Long-edge pixel height used for the FIRST rendering attempt. Not a
#: threshold and not a guarantee — see the module docstring. It is simply
#: where the ladder starts.
SURYA_PAGE_HEIGHT = 1770

#: Renderings tried in order until one produces text. Each entry is
#: ``(long_edge_height, cv2_interpolation_name)``.
#:
#: ORDERED BY MEASUREMENT, on one soft page, with thin evidence: 2 of 12
#: renderings recovered it, and both winners were at height 1770 (with the
#: ``cubic`` and ``area`` filters), while every rendering at 1755 and at 1800
#: failed. The crisp page read on all twelve, so the ladder costs it nothing —
#: it stops at rung one.
#:
#: Re-derive this order when more soft pages are available. The ladder's VALUE
#: is that a single rendering's failure is not the page's failure; the specific
#: rungs are the replaceable part.
SURYA_RENDER_LADDER = (
    (1770, "cubic"),
    (1770, "area"),
    (1770, "lanczos4"),
    (1755, "cubic"),
    (1800, "cubic"),
    (1740, "area"),
)

#: Default endpoint of a `llama-server` started with the Surya GGUF + mmproj.
SURYA_DEFAULT_ENDPOINT = "http://127.0.0.1:18081/v1/chat/completions"

#: The instruction that produced text on every readable page measured.
SURYA_PROMPT = "Read all the text in this document."

_STATUS_TEXT = "text"
_STATUS_LAYOUT_ONLY = "layout-only"
_STATUS_EMPTY = "empty"
_STATUSES = (_STATUS_TEXT, _STATUS_LAYOUT_ONLY, _STATUS_EMPTY)


@dataclass(frozen=True)
class SuryaReading:
    """What the eyes saw — always this shape, never a bare string.

    ``status`` is the three-valued signal the caller must branch on:

    ``"text"``
        Characters were recovered; ``text`` holds them.
    ``"layout-only"``
        The model answered with block geometry and no characters. This is a
        FAILURE, not a page without text — see the module docstring. ``text``
        is ``None``, because "" would be indistinguishable from a genuinely
        blank page.
    ``"empty"``
        The model returned nothing usable at all.

    ``sent_size`` is the ``(width, height)`` actually transmitted after
    normalization, which is rarely the source size and is the number a caller
    needs in order to reason about a failure. No field names a *cause*: this
    engine does not know one, and inventing it would cross into interpretation
    the caller owns.
    """

    status: str
    text: Optional[str]
    body: str
    source_size: Tuple[int, int]
    sent_size: Tuple[int, int]
    image_tokens: Optional[int]
    rendering: Optional[Tuple[int, str]] = None
    attempts: int = 1

    def __post_init__(self) -> None:
        # Fail where the answer is built, not three layers downstream.
        if self.status not in _STATUSES:
            raise ValueError(
                f"SuryaReading.status must be one of {_STATUSES}, "
                f"got {self.status!r}"
            )
        if self.status == _STATUS_TEXT and not self.text:
            raise ValueError(
                "SuryaReading.status == 'text' requires non-empty text; "
                "a reading with no characters is not a text reading"
            )
        if self.status != _STATUS_TEXT and self.text is not None:
            raise ValueError(
                f"SuryaReading.text must be None when status is "
                f"{self.status!r} — an empty string would be "
                "indistinguishable from a genuinely blank page"
            )

    @property
    def read(self) -> bool:
        """True only when characters were actually recovered."""
        return self.status == _STATUS_TEXT


def _interpolation(name: str) -> int:
    import cv2

    table = {
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
        "linear": cv2.INTER_LINEAR,
    }
    try:
        return table[name]
    except KeyError:
        raise ValueError(
            f"unknown interpolation {name!r}; expected one of {sorted(table)}"
        ) from None


def normalize_page(
    image: np.ndarray,
    height: int = SURYA_PAGE_HEIGHT,
    filter_name: str = "cubic",
) -> np.ndarray:
    """Resample ``image`` so its long edge is ``height``, preserving aspect.

    Scales DOWN as readily as up — the point is to land on a rendering the
    model reads, not to maximise pixels. Upscaling a soft scan too far was
    measured to break recognition that worked at a smaller size.

    ``filter_name`` matters and is not cosmetic: on the soft scan, ``cubic``
    and ``area`` at height 1770 recovered the text while ``lanczos4`` and
    ``linear`` at the same size did not. This is why :func:`ocr_surya` varies
    it across the ladder rather than fixing one filter.
    """
    if image.ndim < 2:
        raise ValueError(
            f"normalize_page expects a 2-D or 3-D image array, got shape {image.shape}"
        )
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"normalize_page got a degenerate image: {src_w}x{src_h}")

    interp = _interpolation(filter_name)
    long_edge = max(src_h, src_w)
    if long_edge == height:
        return image

    import cv2

    scale = height / float(long_edge)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    return cv2.resize(image, (dst_w, dst_h), interpolation=interp)


def is_layout_only(body: str) -> bool:
    """True when the model returned block geometry instead of characters.

    Mechanically decidable, which is the whole reason this failure is worth
    detecting: the response is a JSON array of ``{label, bbox, count}`` objects
    carrying no text, and it arrives with ``finish_reason="stop"`` and a
    plausible byte count. It reads as success to anything that only checks that
    the request completed.
    """
    stripped = body.strip()
    if not stripped.startswith("["):
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, list) or not parsed:
        return False
    return all(isinstance(item, dict) and "bbox" in item for item in parsed)


def _encode_png(image: np.ndarray) -> str:
    import cv2

    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to PNG-encode the image for the Surya endpoint")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, Path)):
        import cv2

        loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"could not read an image from {image!r}")
        return loaded
    raise TypeError(
        "ocr_surya(image): image must be a file path (str/Path) or a numpy "
        f"array, got {type(image).__name__}"
    )


def _ask(
    prepared: np.ndarray,
    endpoint: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> Tuple[str, Optional[int]]:
    """POST one rendering and return ``(body, image_tokens)``."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + _encode_png(prepared)
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed: Any = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        raise ConnectionError(
            f"Surya endpoint {endpoint} is unreachable ({exc}). Start it with: "
            "llama-server -m surya-2.gguf --mmproj surya-2-mmproj.gguf "
            "--port 18081 -ngl 99 -c 4096"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConnectionError(
            f"Surya endpoint {endpoint} returned a non-JSON body ({exc}); it is "
            "probably not a llama-server OpenAI-compatible endpoint"
        ) from exc

    try:
        body = parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ConnectionError(
            f"Surya endpoint {endpoint} returned JSON without "
            f"choices[0].message.content: {parsed!r}"
        ) from exc

    image_tokens = None
    usage = parsed.get("usage")
    if isinstance(usage, dict):
        image_tokens = usage.get("prompt_tokens")

    return body, image_tokens


def _classify(body: str) -> str:
    if not body or not body.strip():
        return _STATUS_EMPTY
    if is_layout_only(body):
        return _STATUS_LAYOUT_ONLY
    return _STATUS_TEXT


def ocr_surya(
    image: Union[str, Path, np.ndarray],
    endpoint: str = SURYA_DEFAULT_ENDPOINT,
    prompt: str = SURYA_PROMPT,
    max_tokens: int = 4096,
    timeout: float = 300.0,
    ladder: Tuple[Tuple[int, str], ...] = SURYA_RENDER_LADDER,
) -> SuryaReading:
    """Read ``image`` via a running Surya ``llama-server``.

    Walks ``ladder``, stopping at the first rendering whose response is not
    layout-only. A single rendering's failure is NOT the page's failure — 2 of
    12 renderings recovered a soft scan that failed at its native size — so
    giving up after one attempt would discard readable pages. A crisp page
    reads on the first rung and costs nothing extra (measured 12 of 12).

    Pass ``ladder=((height, filter),)`` for a single attempt, or
    ``ladder=()`` to send the source image untouched.

    Returns a :class:`SuryaReading` in every non-transport case, including
    when no rung read the page — the failure is data, not an exception,
    because "this page produced no characters" is a fact the caller routinely
    needs to record rather than crash on. ``rendering`` names the rung that
    was used and ``attempts`` how many were tried.

    Raises
    ------
    TypeError
        If ``image`` is neither path-like nor a numpy array.
    ValueError
        If a ladder rung names an unknown interpolation filter.
    ConnectionError
        If the endpoint could not be reached or answered malformed JSON. A
        transport failure genuinely is exceptional: it says nothing about the
        page, so returning a reading would misattribute it to the document.
    """
    source = _load_image(image)
    src_h, src_w = source.shape[:2]

    rungs: Tuple[Optional[Tuple[int, str]], ...] = tuple(ladder) or (None,)

    last: Optional[SuryaReading] = None
    for attempt, rung in enumerate(rungs, start=1):
        prepared = (
            source if rung is None else normalize_page(source, rung[0], rung[1])
        )
        sent_h, sent_w = prepared.shape[:2]
        body, image_tokens = _ask(prepared, endpoint, prompt, max_tokens, timeout)
        status = _classify(body)
        last = SuryaReading(
            status=status,
            text=body if status == _STATUS_TEXT else None,
            body=body,
            source_size=(src_w, src_h),
            sent_size=(sent_w, sent_h),
            image_tokens=image_tokens,
            rendering=rung,
            attempts=attempt,
        )
        if status == _STATUS_TEXT:
            return last

    # Every rung produced a non-text answer. Report the last one, which
    # carries the full attempt count — the caller can see the page was not
    # dismissed on a single rendering.
    assert last is not None  # rungs is never empty
    return last


__all__ = [
    "SURYA_DEFAULT_ENDPOINT",
    "SURYA_PAGE_HEIGHT",
    "SURYA_PROMPT",
    "SURYA_RENDER_LADDER",
    "SuryaReading",
    "is_layout_only",
    "normalize_page",
    "ocr_surya",
]

# EOF
