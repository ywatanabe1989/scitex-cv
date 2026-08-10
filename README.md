# scitex-cv

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center"><b>Small cv2-based image utilities — I/O, transform, filters, drawing.</b></p>

<p align="center">
  <a href="https://scitex-cv.readthedocs.io/">Full Documentation</a> · <code>uv pip install scitex-cv[all]</code>
</p>

<!-- scitex-badges:start -->
<p align="center">
  <a href="https://pypi.org/project/scitex-cv/"><img src="https://img.shields.io/pypi/v/scitex-cv.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/scitex-cv/"><img src="https://img.shields.io/pypi/pyversions/scitex-cv.svg" alt="Python"></a>
  <a href="https://github.com/ywatanabe1989/scitex-cv/actions/workflows/test.yml"><img src="https://github.com/ywatanabe1989/scitex-cv/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/ywatanabe1989/scitex-cv"><img src="https://codecov.io/gh/ywatanabe1989/scitex-cv/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scitex-cv.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/scitex-cv/badge/?version=latest" alt="Docs"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
</p>
<!-- scitex-badges:end -->

---

## Installation

```bash
pip install scitex-cv
```

## Architecture

```
scitex_cv/
├── __init__.py        ← public API (load, save, resize, blur, edge_detect, ...)
├── _io.py             ← image I/O wrappers (PIL / OpenCV backends)
├── _transform.py      ← geometric transforms (resize, crop, rotate)
├── _filters.py        ← blur, sharpen, edge-detect filters
├── _draw.py           ← annotation helpers (draw boxes, text, masks)
└── _ocr.py            ← image → text (EasyOCR; optional `ocr` extra)
```

Thin, opinionated wrapper around PIL + OpenCV — every public name in
`__init__.py` re-exports from one of the five leaf modules above.

## Quick Start

```python
import scitex_cv as cv

img = cv.load("input.png")
img = cv.resize(img, scale=0.5)
img = cv.blur(img, ksize=5)
edges = cv.edge_detect(img, method="canny")
cv.save(edges, "edges.png")
```

## 1 Interfaces

<details open>
<summary><strong>Python API</strong></summary>

<br>

```python
import scitex_cv as cv

# I/O
img = cv.load("input.png")
cv.save(img, "out.png")
gray = cv.to_gray(img); rgb = cv.to_rgb(img); bgr = cv.to_bgr(img)

# Transform
cv.resize(img, scale=0.5)
cv.rotate(img, angle=90)
cv.flip(img, direction="horizontal")
cv.crop(img, x=10, y=10, width=100, height=100)
cv.pad(img, top=10, bottom=10, left=10, right=10)

# Filters
cv.blur(img, ksize=5); cv.sharpen(img)
cv.edge_detect(img, method="canny")
cv.threshold(img, thresh=128); cv.denoise(img)

# Drawing
cv.rectangle(img, (5, 5), (30, 30), color=(0, 255, 0))
cv.circle(img, (50, 50), radius=10)
cv.line(img, (0, 0), (100, 100))
cv.arrow(img, (0, 0), (50, 50))
cv.text(img, "label", (10, 30))
cv.polylines(img, points=np.array([[0, 0], [100, 0], [100, 100]]))

# OCR (image → text; requires the optional `ocr` extra: pip install 'scitex-cv[ocr]')
cv.ocr("scan.png")                       # → recognized text (str)
cv.ocr(img, languages=("en",))           # in-memory array, English only
cv.ocr(img, detail=True)                 # → [(bbox, text, confidence), ...]
```

</details>

## Demo

```mermaid
flowchart LR
    F["input.png"] --> L["cv.load()"]
    L --> R["cv.resize(scale=0.5)"]
    R --> B["cv.blur(ksize=5)"]
    B --> E["cv.edge_detect(method='canny')"]
    E --> S["cv.save('edges.png')"]
    S --> O["edges.png"]
```

## Status

Standalone fork of `scitex.cv`. Only deps are numpy + opencv-python.
The umbrella package's `scitex.cv` import path is preserved via a
`sys.modules`-alias bridge.

## Part of SciTeX

`scitex-cv` is part of [**SciTeX**](https://scitex.ai). Install via
the umbrella with `pip install scitex[cv]` to use as
`scitex.cv` (Python) or `scitex cv ...` (CLI).

>Four Freedoms for Research
>
>0. The freedom to **run** your research anywhere — your machine, your terms.
>1. The freedom to **study** how every step works — from raw data to final manuscript.
>2. The freedom to **redistribute** your workflows, not just your papers.
>3. The freedom to **modify** any module and share improvements with the community.
>
>AGPL-3.0 — because we believe research infrastructure deserves the same freedoms as the software it runs on.

## License

AGPL-3.0-only (see [LICENSE](./LICENSE)).

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
</p>
