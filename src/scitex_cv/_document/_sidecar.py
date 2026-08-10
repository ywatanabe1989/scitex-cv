#!/usr/bin/env python3
# Timestamp: 2026-08-10
# File: src/scitex_cv/_document/_sidecar.py
"""The on-disk result beside the input, and the policy for resuming from it.

The consuming layer must be able to read a result without calling Python, and
a partial run must resume. The operator's flow is "scan a stack, let it run
overnight", so a crash at page 300 must not restart at page 1.

The ladder fingerprint, and why a stale negative is the dangerous one
--------------------------------------------------------------------
The sidecar records the ladder it was produced under. On resume:

* ``text`` and ``blank`` are kept — they are settled facts about the page.
* ``error`` is ALWAYS retried — the failure described the run, not the page,
  so carrying it forward would turn a transient outage into a permanent verdict.
* ``unreadable`` is kept ONLY under an unchanged ladder. Under a different
  ladder the page may well read, and a stored "cannot be read" that outlives
  the renderings that produced it is a negative result frozen past its
  evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ._types import STATUS_ERROR, STATUS_UNREADABLE, DocumentReading, PageReading

#: Suffix appended to the input path to locate its sidecar. Stable by design:
#: the consumer must find a result by name, without running code.
SIDECAR_SUFFIX = ".scitex-cv-ocr.json"

#: Schema version. Bump when the on-disk shape changes in a way an older
#: reader would MISINTERPRET (not merely fail to use). A reader finding a
#: higher version refuses rather than guessing.
SIDECAR_VERSION = 1


def sidecar_path_for(document_path: Union[str, Path]) -> Path:
    """Where the sidecar for ``document_path`` lives. Stable and adjacent."""
    path = Path(document_path)
    return path.with_name(path.name + SIDECAR_SUFFIX)


def ladder_fingerprint(ladder: Tuple[Tuple[int, str], ...]) -> str:
    """Stable identifier for a ladder, recorded in the sidecar.

    Used on resume to decide whether a stored ``unreadable`` verdict still
    applies — it was produced by one specific sequence of renderings.
    """
    return ";".join(f"{height}:{filter_name}" for height, filter_name in ladder)


def read_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    """Load a sidecar, or None when there is no usable prior result.

    A corrupt or unreadable sidecar returns None — that is a reason to redo
    the work, not to lose the run. A sidecar from a NEWER writer raises,
    because silently reinterpreting an unknown shape is how a wrong result
    looks like a right one.
    """
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("sidecar_version")
    if not isinstance(version, int) or version > SIDECAR_VERSION:
        raise ValueError(
            f"{path} was written by sidecar_version {version!r}, but this "
            f"reader understands at most {SIDECAR_VERSION}. Upgrade scitex-cv, "
            "or delete the sidecar to redo the document from scratch."
        )
    return payload


def page_from_payload(payload: Dict[str, Any]) -> Optional[PageReading]:
    """Rebuild one page entry, or None when it is malformed.

    One bad entry invalidates that page only — the rest of the document's
    completed work is still worth keeping.
    """
    try:
        rendering = payload.get("rendering")
        return PageReading(
            index=payload["index"],
            status=payload["status"],
            text=payload.get("text"),
            source=payload.get("source"),
            rendering=tuple(rendering) if rendering else None,
            image_tokens=payload.get("image_tokens"),
            attempts=payload.get("attempts", 0),
            detail=payload.get("detail"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def reusable(
    page: PageReading, stored_fingerprint: str, current_fingerprint: str
) -> bool:
    """Whether a stored page result may be carried forward untouched."""
    if page.status == STATUS_ERROR:
        return False
    if page.status == STATUS_UNREADABLE:
        return stored_fingerprint == current_fingerprint
    return True


def write_sidecar(path: Path, reading: DocumentReading) -> None:
    """Write the sidecar ATOMICALLY, so a crash cannot leave a truncated one.

    Written to a ``.partial`` sibling, fsynced, then renamed — os.replace is
    atomic within a filesystem. A sidecar that exists is therefore always
    complete and parseable, which is what lets a resume trust it.
    """
    payload = {
        "sidecar_version": SIDECAR_VERSION,
        "document": reading.path,
        "page_count": reading.page_count,
        "ladder": [list(rung) for rung in reading.ladder],
        "ladder_fingerprint": ladder_fingerprint(reading.ladder),
        "pages": [
            {
                key: (list(value) if key == "rendering" and value else value)
                for key, value in asdict(page).items()
            }
            for page in reading.pages
        ],
    }
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# EOF
