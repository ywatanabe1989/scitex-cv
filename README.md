# scitex-cv

Small cv2/Pillow image-processing utilities, extracted from the [SciTeX](https://github.com/ywatanabe1989/scitex-python) ecosystem as a standalone package.

## Install

```bash
pip install scitex-cv
```

## API

```python
import scitex_cv as cv

# I/O
img = cv.load("input.png")
cv.save(img, "out.png")
gray = cv.to_gray(img); rgb = cv.to_rgb(img); bgr = cv.to_bgr(img)

# Transform
cv.resize(img, scale=0.5)
cv.rotate(img, 90); cv.flip(img, axis="horizontal")
cv.crop(img, x=10, y=10, w=100, h=100)
cv.pad(img, top=10, bottom=10, left=10, right=10)

# Filters
cv.blur(img, ksize=5); cv.sharpen(img)
cv.edge_detect(img, method="canny")
cv.threshold(img, value=128); cv.denoise(img)

# Drawing
cv.rectangle(img, (5,5), (30,30), color=(0,255,0))
cv.circle(img, (50,50), radius=10)
cv.line(img, (0,0), (100,100))
cv.arrow(img, (0,0), (50,50))
cv.text(img, "label", (10,30))
cv.polylines(img, points=[(0,0),(100,0),(100,100)])
```

## Status

Standalone fork of `scitex.cv`. Only deps are numpy + opencv-python + Pillow.
The umbrella package's `scitex.cv` import path is preserved via a
`sys.modules`-alias bridge.

## License

AGPL-3.0-only (see [LICENSE](./LICENSE)).
