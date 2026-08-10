---
description: |
  [TOPIC] Python API
  [DETAILS] Public Python API of scitex-cv — exported functions, signatures,
  return types, and minimal usage examples per function.
tags: [scitex-cv-python-api]
---

# Python API

Import via:

```python
import scitex_cv as cv
```

All public names are re-exported from `scitex_cv.__init__`.

## I/O (`_io.py`)

```python
img = cv.load("input.png", color=True, alpha=False)
cv.save(img, "out.png", quality=95)
gray = cv.to_gray(img)
rgb  = cv.to_rgb(img)
bgr  = cv.to_bgr(img)
```

| Function | Returns | Description |
|---|---|---|
| `load(path, color, alpha)` | `np.ndarray` | Read image from disk (BGR). Raises `FileNotFoundError` on failure. |
| `save(img, path, quality)` | `Path` | Write image to disk; parent dirs created on demand. |
| `to_gray(img)` | `np.ndarray` | Convert BGR/RGB → grayscale. |
| `to_rgb(img)` | `np.ndarray` | Convert BGR → RGB. Handles alpha. |
| `to_bgr(img)` | `np.ndarray` | Convert RGB → BGR. Handles alpha. |

## Transform (`_transform.py`)

```python
r1 = cv.resize(img, size=(320, 240))
r2 = cv.resize(img, scale=0.5)
rot = cv.rotate(img, angle=90)
flp = cv.flip(img, direction="horizontal")
crp = cv.crop(img, x=10, y=10, width=100, height=100)
pad = cv.pad(img, top=10, bottom=10, left=10, right=10)
```

| Function | Returns | Description |
|---|---|---|
| `resize(img, size, scale, interpolation)` | `np.ndarray` | Resize by target `(w, h)` or `scale` factor. Interpolation: nearest, linear, cubic, area, lanczos. |
| `rotate(img, angle, center, scale)` | `np.ndarray` | Rotate counter-clockwise by degrees. |
| `flip(img, direction)` | `np.ndarray` | Flip horizontal, vertical, or both. |
| `crop(img, x, y, width, height)` | `np.ndarray` | Extract a rectangular region (returns a copy). |
| `pad(img, top, bottom, left, right, color, mode)` | `np.ndarray` | Add borders. Modes: constant, reflect, replicate. |

## Filters (`_filters.py`)

```python
b1 = cv.blur(img, ksize=5, method="gaussian")
sh = cv.sharpen(img, strength=1.0)
ed = cv.edge_detect(img, method="canny", low=50, high=150)
th = cv.threshold(img, thresh=128, method="binary")
dn = cv.denoise(img, strength=10, method="fastNl")
```

| Function | Returns | Description |
|---|---|---|
| `blur(img, ksize, method)` | `np.ndarray` | Blur: gaussian, median, box, or bilateral. ksize auto-rounded to odd. |
| `sharpen(img, strength)` | `np.ndarray` | Unsharp-mask–style sharpen. |
| `edge_detect(img, method, low, high)` | `np.ndarray` | Edge detection: canny, sobel, or laplacian. |
| `threshold(img, thresh, maxval, method)` | `np.ndarray` | Binarise: binary, binary_inv, trunc, tozero, otsu, adaptive_mean, adaptive_gaussian. |
| `denoise(img, strength, method)` | `np.ndarray` | Denoise: fastNl or bilateral. |

## Drawing (`_draw.py`)

```python
cv.rectangle(img, (5, 5), (100, 100), color=(0, 255, 0), thickness=2, filled=False)
cv.circle(img, (50, 50), radius=10, color=(0, 255, 0))
cv.line(img, (0, 0), (100, 100), color=(0, 255, 0))
cv.arrow(img, (0, 0), (50, 50), color=(0, 255, 0), tip_length=0.1)
cv.text(img, "label", (10, 30), scale=1.0, font="simplex")
cv.polylines(img, pts, closed=True, color=(0, 255, 0))
```

| Function | Returns | Description |
|---|---|---|
| `rectangle(img, pt1, pt2, color, thickness, filled)` | `np.ndarray` | Draw a rectangle (in-place). |
| `circle(img, center, radius, color, thickness, filled)` | `np.ndarray` | Draw a circle (in-place). |
| `line(img, pt1, pt2, color, thickness)` | `np.ndarray` | Draw a line (in-place). |
| `arrow(img, pt1, pt2, color, thickness, tip_length)` | `np.ndarray` | Draw an arrowed line (in-place). |
| `text(img, text, position, color, scale, thickness, font)` | `np.ndarray` | Draw text (in-place). Fonts: simplex, plain, duplex, complex, triplex. |
| `polylines(img, points, closed, color, thickness)` | `np.ndarray` | Draw polylines from an (N, 2) array (in-place). |
