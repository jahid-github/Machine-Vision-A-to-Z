# HAMK Robotics Machine Vision Final Exam Study Guide

This consolidated README combines both of your drafts into one clean, non-redundant study guide. It covers all five lecture modules with key theory, formulas, and practical Python/OpenCV snippets.

## How To Use This Guide

1. Read each module in order (1 -> 5).
2. Practice the code snippets without copying.
3. Review the final quick-reference pipeline before the exam.

## Module 1: Introduction to Machine Vision and Tools

Machine vision enables robots to convert visual data into actionable decisions.

### Core Concepts

- Images are matrices.
- Grayscale image: 2D matrix, pixel values in `0..255`.
- Color image in OpenCV: 3D matrix `(height, width, 3)` using **BGR** order.
- Image coordinates: origin `(0, 0)` is top-left, `u` increases to the right, `v` increases downward.

### RoboDK + OpenCV Snapshot Workflow

```python
from robodk.robolink import *
import cv2

RDK = Robolink()
cam = RDK.Item('Camera 1')

file = 'snapshot.png'
RDK.Cam2D_Snapshot(file, cam)
img = cv2.imread(file)
```

## Module 2: Low-Level Vision I - Filtering and Enhancement

Preprocessing improves image quality before segmentation and measurement.

### Point Operations

A pixel is modified based only on its original value.

- Brightness/contrast model:

```text
g(x, y) = alpha * f(x, y) + beta
```

- `alpha` controls contrast, `beta` controls brightness.

### Histogram and Histogram Equalization

- Histogram shows pixel-intensity distribution.
- Histogram equalization improves contrast by spreading frequent intensity values.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)
```

### Spatial Filtering (Convolution)

A kernel slides over the image and computes local weighted sums.

### Smoothing Filters

- Mean/Average: simple blur, noise reduction.
- Gaussian: weighted blur, strong for random (Gaussian) noise.
- Median: best for salt-and-pepper noise.

```python
mean = cv2.blur(img, (5, 5))
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
```

## Module 3: Low-Level Vision II - Segmentation, Edges, Morphology

Goal: separate object from background and refine mask quality.

### Thresholding

- Global threshold: one fixed threshold for whole image.
- Otsu threshold: automatically picks threshold by minimizing intra-class variance.
- Adaptive threshold: local thresholds for uneven lighting.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, global_bin = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
_, otsu_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive_bin = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)
```

### Edge Detection

- Sobel: gradient-based edge response.
- Canny: robust multi-stage detector (industry standard).

```python
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
edges = cv2.Canny(blur, 100, 200)
```

### Morphological Operations

Use binary images with a structuring element (`kernel`).

- Erosion: shrinks white regions (removes small white noise).
- Dilation: expands white regions.
- Opening: erosion -> dilation (noise removal).
- Closing: dilation -> erosion (fills small holes).

```python
kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(otsu_bin, kernel, iterations=1)
dilated = cv2.dilate(otsu_bin, kernel, iterations=1)
opened = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, kernel)
```

## Module 4: Coordinate Mapping (Calibration)

This module maps camera pixels to robot coordinates.

### Why It Matters

- Camera gives `(u, v)` in pixels.
- Robot needs `(X, Y, Z)` in mm.

### 2D Transformations

- Translation: `(x + dx, y + dy)`.
- Rigid (Euclidean): rotation + translation.
- Affine: rotation + translation + scale + shear.
- Homography: projective planar mapping.

### Homography Essentials

- Uses a `3x3` matrix `H`.
- Requires at least **4 point pairs**.
- Maps points on one plane to another plane.

```text
s * [X, Y, 1]^T = H * [u, v, 1]^T
```

### Calibration Workflow

1. Collect 4+ pixel points from the camera image.
2. Collect corresponding robot/world points in mm.
3. Compute `H` using `cv2.findHomography()`.
4. Transform new detections with `cv2.perspectiveTransform()`.

```python
import numpy as np
import cv2

# Example format: Nx2 arrays with matching order
src_pixels = np.array([
    [120, 90],
    [420, 85],
    [430, 310],
    [110, 320]
], dtype=np.float32)

dst_robot = np.array([
    [100, 200],
    [300, 200],
    [300, 50],
    [100, 50]
], dtype=np.float32)

H, mask = cv2.findHomography(src_pixels, dst_robot)

target_pixel = np.array([[[u, v]]], dtype=np.float32)
robot_coord = cv2.perspectiveTransform(target_pixel, H)
```

## Module 5: Color and Shape Analysis

Combine color segmentation with geometric descriptors for robust object identification.

### HSV Color Space

- `H` (Hue): color (`0..179` in OpenCV).
- `S` (Saturation): color intensity.
- `V` (Value): brightness.

Why HSV instead of BGR in industry:
- HSV separates color (`H`) from illumination (`V`), so color detection is more stable under lighting changes.

### Color Masking Example

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Red often needs two hue ranges because hue wraps around
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
```

### Geometric Descriptors

- Aspect ratio:

```text
aspect_ratio = width / height
```

- Circularity:

```text
circularity = 4 * pi * (area / perimeter^2)
```

A perfect circle has circularity `1.0`.

```python
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    circularity = 4 * np.pi * area / (perimeter ** 2)
```

### SimpleBlobDetector (Optional Alternative)

```python
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.75

blob_detector = cv2.SimpleBlobDetector_create(params)
keypoints = blob_detector.detect(red_mask)
```

## Final Exam Quick-Reference Pipeline

```python
import cv2
import numpy as np

# 1) Acquire image
img = cv2.imread('snapshot.png')

# 2) Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3) Segmentation (Otsu + morphology)
_, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 4) Color detection (HSV)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
red_mask = cv2.inRange(hsv, lower_red, upper_red)

# 5) Coordinate mapping (homography)
# src_pixels: [[u1, v1], [u2, v2], ...]
# dst_robot: [[x1, y1], [x2, y2], ...]
H, _ = cv2.findHomography(src_pixels, dst_robot)
target_pixel = np.array([[[u, v]]], dtype=np.float32)
robot_coord = cv2.perspectiveTransform(target_pixel, H)
```

## Final Exam Checklist

- Explain why OpenCV uses BGR and when to convert to grayscale or HSV.
- Choose the right smoothing filter for a given noise type.
- Compare global, Otsu, and adaptive thresholding.
- Explain erosion, dilation, opening, and closing with practical use cases.
- Explain Sobel vs Canny and when Canny is preferred.
- Describe translation, rigid, affine, and homography transforms.
- State that homography needs at least 4 corresponding planar points.
- Compute and interpret aspect ratio and circularity.
- Explain why HSV is usually more stable than BGR in industrial lighting.

## Pro Tip for Theory Questions

If asked why HSV is preferred over BGR/RGB:
- HSV decouples color information (`Hue`) from illumination (`Value`).
- This makes threshold-based color detection significantly more robust to shadows and brightness changes.
