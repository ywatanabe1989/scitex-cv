---
name: scitex-cv
description: |
  [WHAT] Computer vision helpers for SciTeX research scripts — image I/O, batched preprocessing, and lightweight model wrappers.
  [WHEN] Loading and preparing image datasets for analysis or modeling.
  [HOW] `from scitex_cv import ...` or `scitex-cv --help`.
primary_interface: python
interfaces:
  python: 2
  cli: 0
  mcp: 0
  skills: 2
  hook: 0
  http: 0
canonical-location: scitex-cv/src/scitex_cv/_skills/scitex-cv/SKILL.md
tags: [scitex-cv]
---

> **Interfaces:** Python ⭐⭐ · CLI — · MCP — · Skills ⭐⭐ · Hook — · HTTP —

# scitex-cv

Small cv2-based image utilities. I/O (load/save/color-convert), transform (resize/rotate/flip/crop/pad), filters (blur/sharpen/edge/threshold/denoise), draw (rectangle/circle/line/text/polylines/arrow). Drop-in replacement for memorising cv2's BGR-vs-RGB swaps and dtype quirks.

See README.md and the package's public `__init__.py` for the full
function list. This skill leaf exists so agents discover the package
exists and roughly what shape it has — refer to the source for
signatures.

## Sub-skills

### Core (01–09)
- [01_installation.md](01_installation.md) — install + import sanity check
- [02_quick-start.md](02_quick-start.md) — 30-second tour
- [03_python-api.md](03_python-api.md) — Python API surface
