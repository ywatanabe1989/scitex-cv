---
description: |
  [TOPIC] Quick Start
  [DETAILS] Smallest useful example demonstrating the primary use case in
  under 30 seconds.
tags: [scitex-cv-quick-start]
---

# Quick Start

```python
import scitex_cv as cv
import numpy as np

# Load
img = cv.load("input.png")

# Transform
resized = cv.resize(img, scale=0.5)
rotated = cv.rotate(img, angle=90)
cropped = cv.crop(img, x=10, y=10, width=100, height=100)

# Filter
blurred = cv.blur(img, ksize=5)
edges = cv.edge_detect(img, method="canny")

# Draw
cv.rectangle(img, (5, 5), (100, 100), color=(0, 255, 0))
cv.text(img, "hello", (10, 30))

# Save
cv.save(edges, "edges.png")
```
