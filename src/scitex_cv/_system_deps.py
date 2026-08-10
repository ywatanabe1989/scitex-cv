#!/usr/bin/env python3
"""OS-level (apt) system dependencies for scitex-cv.

scitex-cv depends on ``opencv-python-headless``, whose manylinux wheel
bundles all its native libraries (ffmpeg, libpng, openblas, libavif, …).
Verified — ``ldd`` on ``cv2*.so`` plus runtime ``/proc/self/maps`` while
exercising every cv2 operation scitex-cv uses — that it loads **no**
X11 / OpenGL / GLib system libraries (no libxcb1, libgl1, or libglib2.0-0).
The only external dependency is ``libz``/zlib1g, part of every base image.
So scitex-cv needs **no** apt packages at the OS level.

This module still registers a ``scitex_dev.system_deps`` provider so the
ecosystem aggregator records that scitex-cv was assessed and requires
nothing — ``provide()`` returns an empty list. (Earlier revisions declared
the X11 libs for full ``opencv-python``; the switch to headless dropped
them.)
"""

from __future__ import annotations

from typing import List

#: Owning leaf package (kept for parity with other leaves / future use).
PROVIDER = "scitex-cv"


def apt_packages() -> List[str]:
    """Apt package names scitex-cv needs — none with opencv-python-headless."""
    return []


def provide() -> list:
    """Entry-point for the ``scitex_dev.system_deps`` group.

    Returns an empty list: the headless OpenCV wheel is self-contained, so
    scitex-cv declares no OS-level apt dependencies.
    """
    return []


__all__ = ["PROVIDER", "apt_packages", "provide"]

# EOF
