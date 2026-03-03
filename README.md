# Machine Vision Exam Master

This README is a complete, exam-focused summary of core machine vision topics: theory, formulas, matrices, segmentation, color, shape, geometry, calibration, and coding patterns.

## 1. Complete Machine Vision Pipeline

1. **Image Acquisition**
   - Camera captures image.
   - Lighting quality is critical.
   - Lens defines field of view.
2. **Preprocessing**
   - Improve contrast.
   - Reduce noise.
   - Prepare for segmentation.
3. **Segmentation**
   - Separate object from background.
   - Output is usually a binary mask (`white = object`, `black = background`).
4. **Feature Extraction**
   - Shape: area, circularity, aspect ratio.
   - Color: HSV-based features.
   - Geometry: position/orientation.
5. **Analysis and Decision**
   - Classification.
   - Measurement.
   - Pass/fail decision.
6. **Action**
   - Robot motion, pick-and-place, sorting, or process control.

## 2. Image Representation

An image is represented as:

\[
f(x, y)
\]

- \(x, y\): spatial coordinates
- \(f(x, y)\): intensity value

In practice:
- Image = matrix
- Pixel = smallest unit
- Grayscale intensity range: `0..255`
- Color image in OpenCV: BGR channels

## 3. Intensity Transformations

### Brightness and Contrast

<img width="754" height="313" alt="image" src="https://github.com/user-attachments/assets/20d2466b-c98a-4aee-b31c-035c2175aec1" />


```python
img_new = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
```

### Linear Contrast Stretching

<img width="916" height="220" alt="image" src="https://github.com/user-attachments/assets/613b3103-ac24-44eb-af32-bacc1cee6181" />


```python
img_new = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
```

## 4. Histogram

- X-axis: intensity (`0..255`)
- Y-axis: pixel count

Used to inspect:
- brightness
- contrast
- under/overexposure
- bimodal behavior (useful for thresholding)

## 5. Noise Types

- Gaussian noise
- Salt-and-pepper noise
- Sensor noise
- Motion blur

## 6. Filtering and Convolution

Convolution process:
1. Place kernel over local neighborhood.
2. Multiply element-wise.
3. Sum results.
4. Write output pixel.

Use cases:
- blurring/smoothing
- edge detection
- sharpening

### Smoothing Filters

- **Mean filter**: simple blur
- **Gaussian filter**: better noise suppression
- **Median filter**: strong for salt-and-pepper noise

```python
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
img_med = cv2.medianBlur(img, 5)
```

## 7. Segmentation

Goal: isolate object from background.  
Typical output: binary image (`0` and `255`).

### Global Threshold

\[
g(x,y)=
\begin{cases}
1, & f(x,y)\ge T \\
0, & f(x,y)<T
\end{cases}
\]

```python
_, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
```

Best when illumination is uniform.

### Otsu Threshold

- Automatic threshold selection.
- Assumes bimodal histogram.
- Maximizes between-class variance.

```python
_, binary = cv2.threshold(
    img, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

### Adaptive Threshold

- Threshold varies per local region.
- Better under uneven lighting.

## 8. Edge Detection

### Sobel

- Gradient-based edge response.

### Canny (high priority)

Pipeline:
1. Gaussian smoothing
2. Gradient computation
3. Non-maximum suppression
4. Double threshold
5. Edge tracking by hysteresis

```python
edges = cv2.Canny(img, 100, 200)
```

## 9. Morphological Operations

```python
kernel = np.ones((3, 3), np.uint8)
```

- **Erosion**: shrinks white regions
  ```python
  eroded = cv2.erode(binary, kernel)
  ```
- **Dilation**: expands white regions
  ```python
  dilated = cv2.dilate(binary, kernel)
  ```
- **Opening** (`erosion -> dilation`): removes small noise
  ```python
  opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
  ```
- **Closing** (`dilation -> erosion`): fills small holes
  ```python
  closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
  ```

## 10. Color Spaces

### Why RGB can fail

RGB mixes brightness and chromatic information, making segmentation fragile under lighting changes.

### HSV

- `H`: Hue (color)
- `S`: Saturation
- `V`: Value/brightness

OpenCV hue range: `0..179`

Typical hue intervals:
- Red: `0..10` and `160..179`
- Green: `50..70`
- Blue: `110..130`

### HSV Segmentation Example

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([110, 50, 50])
upper = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
```

## 11. Contours and Shape Descriptors

Find contours:

```python
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
```

- Area:
  ```python
  area = cv2.contourArea(cnt)
  ```
- Perimeter:
  ```python
  perimeter = cv2.arcLength(cnt, True)
  ```
- Circularity:

\[
C = \frac{4\pi A}{P^2}
\]

If \(C \approx 1\), shape is close to a circle.

- Aspect ratio:
  ```python
  x, y, w, h = cv2.boundingRect(cnt)
  ratio = w / h
  ```

## 12. Coordinate Systems

- **Image coordinates**: pixel domain, origin at top-left, \((u,v)\)
- **Robot/world coordinates**: metric domain (e.g., mm), \((X,Y,Z)\)

Calibration maps image-space to robot-space.

## 13. Homogeneous Coordinates

- Cartesian 2D point: \([x, y]\)
- Homogeneous 2D point: \([x, y, 1]\)

Benefit: translation and other transforms can be represented with matrix multiplication.

## 14. Translation Matrix

\[
T =
\begin{bmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1
\end{bmatrix}
\]

## 15. Rotation Matrix (2D Homogeneous Form)

\[
R =
\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\
\sin\theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
\]

## 16. Rigid Transformation

A rigid transform combines:
- rotation
- translation

It preserves distances and angles.

## 17. Homography (High Priority)

- A \(3 \times 3\) projective transform matrix.
- Maps one plane to another plane.
- Requires at least 4 non-collinear point correspondences.

```python
H, _ = cv2.findHomography(img_pts, robot_pts)
```

Point mapping:

```python
p = np.array([u, v, 1]).reshape(3, 1)
pr = H @ p
pr /= pr[2]
```

## 18. IoU (Intersection over Union)

\[
\mathrm{IoU} = \frac{\text{Intersection}}{\text{Union}}
\]

Used to evaluate segmentation quality against ground truth masks.

## Essential Coding Patterns

### 1) Thresholding Function

```python
def threshold_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(
        img_blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary
```

### 2) HSV Detection Function

```python
def detect_color(path):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([110, 50, 50])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask
```

### 3) Pixel-to-Robot Mapping via Homography

```python
def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1]).reshape(3, 1)
    pr = H @ p
    pr /= pr[2]
    return pr[0:2]
```

---

# MACHINE VISION – FINAL CONFIDENCE TEST (WITH FULL ANSWERS)

---

# SECTION 1 – THEORY

---

## ✅ Q1

**Why apply Gaussian blur before Otsu? What happens if skipped?**

### ✔ Answer:

Gaussian blur reduces noise and small intensity fluctuations in the image.

Otsu’s method assumes a bimodal histogram (object + background).
Noise can create additional peaks in the histogram, making it harder for Otsu to find the correct threshold.

If Gaussian blur is skipped:

* Histogram becomes noisy
* Otsu may select incorrect threshold
* Segmentation becomes unstable

Therefore, smoothing improves threshold reliability.

---

## ✅ Q2

**Why may Otsu fail? Why adaptive works better?**

### ✔ Answer:

Otsu fails when:

* Histogram is not clearly bimodal
* Object and background intensities overlap
* Illumination is uneven

Adaptive threshold works better because:

* It computes threshold locally for each region
* It handles varying lighting conditions
* It adapts to shadows and bright spots

---

## ✅ Q3

**Why RGB fails under shadow but HSV works better?**

### ✔ Answer:

In RGB, brightness and color are mixed.

When a shadow falls:

* R, G, and B values decrease together
* The color range changes
* Thresholding becomes unreliable

In HSV:

* Hue (H) represents color
* Value (V) represents brightness

Shadow mainly affects V, but H remains almost unchanged.

Therefore, HSV segmentation is more robust to lighting variation.

---

## ✅ Q4

**Prove circularity of circle equals 1**

Circularity:

[
C = \frac{4\pi A}{P^2}
]

For circle:
[
A = \pi r^2
]
[
P = 2\pi r
]

Substitute:

[
C = \frac{4\pi(\pi r^2)}{(2\pi r)^2}
]

[
= \frac{4\pi^2 r^2}{4\pi^2 r^2}
= 1
]

Thus, circularity of perfect circle is 1.

---

## ✅ Q5

**Difference between transformations**

### Translation

* Moves object
* Shape unchanged
* Only shifts position

### Rigid Transformation

* Rotation + translation
* Preserves distance and angles

### Affine Transformation

* Includes scaling and shear
* Parallel lines remain parallel

### Homography

* Projective transformation
* Parallel lines may converge
* Used for perspective correction

---

# SECTION 2 – CODING

---

## ✅ Q6

**Threshold + Opening**

```python
import cv2
import numpy as np

def segment_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Smooth
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # Otsu
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    return cleaned
```

---

## ✅ Q7

**Detect Red Circular Objects**

```python
import cv2
import numpy as np

def detect_red_circles(path):

    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red wraps around
    lower1 = np.array([0,50,50])
    upper1 = np.array([10,255,255])
    lower2 = np.array([160,50,50])
    upper2 = np.array([179,255,255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 + mask2

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = 4*np.pi*area/(perimeter**2)

        if circularity > 0.8:
            print("Red circular object detected")
```

---

## ✅ Q8

**Homography Mapping**

```python
import numpy as np
import cv2

H, _ = cv2.findHomography(img_pts, robot_pts)

def pixel_to_robot(u, v, H):

    p = np.array([u, v, 1]).reshape(3,1)

    pr = H @ p

    pr /= pr[2]  # Normalize

    return pr[0], pr[1]
```

---

# SECTION 3 – GEOMETRY

---

## ✅ Q9

Given:

[
R =
\begin{bmatrix}
0 & -1 & 0 \
1 & 0 & 0 \
0 & 0 & 1
\end{bmatrix}
]

Compare with:

[
R =
\begin{bmatrix}
\cos\theta & -\sin\theta \
\sin\theta & \cos\theta
\end{bmatrix}
]

Here:

* cosθ = 0
* sinθ = 1

Thus:

[
\theta = 90^\circ
]

Rotation is counter-clockwise 90°.

---

## ✅ Q10

**Why homogeneous coordinates?**

In Cartesian coordinates, translation cannot be represented as matrix multiplication.

Homogeneous coordinates add an extra dimension:

[
[x, y, 1]
]

This allows translation to be represented as:

[
T =
\begin{bmatrix}
1 & 0 & t_x \
0 & 1 & t_y \
0 & 0 & 1
\end{bmatrix}
]

Thus, translation becomes linear matrix operation.

---

## ✅ Q11

**Robot misses object after calibration. Causes:**

1. Incorrect point selection
2. Inaccurate robot measurements
3. Lens distortion
4. Calibration plane not flat
5. Noise in image detection
6. Wrong coordinate reference frame
7. Not enough calibration points

---

Tell me which one.
