
### **The Machine Vision Pipeline**
*Most problems follow this flow: Capture → Preprocess → Segment → Analyze → Map.*

```python
import cv2
import numpy as np
from robodk.robolink import *

# 1. SETUP & CAPTURE (Lecture 1)
RDK = Robolink()
cam = RDK.Item('Camera 1')
file = "snap.png"
RDK.Cam2D_Snapshot(file, cam)
img = cv2.imread(file) # BGR Format

# 2. PREPROCESSING (Lecture 2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Noise: Use Gaussian for random, Median for salt-and-pepper
blur = cv2.GaussianBlur(gray, (5,5), 0) 
median = cv2.medianBlur(gray, 5)
# Contrast: Histogram Equalization
equ = cv2.equalizeHist(gray)

# 3. SEGMENTATION & EDGES (Lecture 3)
# Otsu (Auto-threshold)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Canny Edges
edges = cv2.Canny(blur, 100, 200)
# Morphology: Open (removes noise), Close (fills holes)
kernel = np.ones((5,5), np.uint8)
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 4. COLOR ANALYSIS (Lecture 5)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([H_min, S_min, V_min]) # Red ~ [0, 100, 100]
upper = np.array([H_max, S_max, V_max]) # Blue ~ [100, 150, 50]
color_mask = cv2.inRange(hsv, lower, upper)

# 5. SHAPE & BLOB DETECTION (Lecture 5)
params = cv2.SimpleBlobDetector_Params()
params.filterByCircularity = True
params.minCircularity = 0.8 # 1.0 = perfect circle
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(clean) # Returns list of (x,y)

# 6. COORDINATE MAPPING (Lecture 4)
# Points: 4 pairs of [u,v] (pixels) and [X,Y] (robot mm)
H, _ = cv2.findHomography(pixel_pts, robot_pts)
# Convert a detected pixel [u,v] to Robot [X,Y]
p = np.array([[[u, v]]], dtype='float32')
robot_coord = cv2.perspectiveTransform(p, H)

```

---

### **Essential Theory Flashcards**

| Topic | Key Concept | Why? |
| --- | --- | --- |
| **Color** | **HSV over BGR** | HSV is robust to lighting; Hue (0-179) is the pure color. |
| **Noise** | **Median Filter** | Best for "Salt & Pepper" noise (outliers). |
| **Threshold** | **Otsu vs Adaptive** | Otsu is global (best for clean BG); Adaptive is local (best for shadows). |
| **Morphology** | **Opening / Closing** | **Opening** = Erode then Dilate (Clean noise). **Closing** = Dilate then Erode (Fill holes). |
| **Mapping** | **Homography** | 8 Degrees of Freedom. Needs **4 points** min. Links Pixels to Robot MM. |
| **Shape** | **Circularity** | $4\pi \times (Area / Perimeter^2)$. If $\approx 1$, it's a circle. |

### **Calculations to Memorize**

1. **Image Shape**: `img.shape` returns `(height, width, channels)`.
2. **Pixel to Robot**: $P_{robot} = H \times P_{pixel}$.
3. **Aspect Ratio**: $Width / Height$.

**Tip 1:** If the code doesn't work, check your **color spaces**. Most errors come from trying to apply `Canny` or `Threshold` on a 3-channel BGR image instead of a 1-channel Grayscale image. Always use `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` first!  

**Tip 2:** Remember that `cv2.imshow()` is for a quick look at an image, but `plt.plot()` or `plt.hist()` is used for analyzing the data behind the pixels to make automated decisions. 

### Plot code

```python
import matplotlib.pyplot as plt
# To show a histogram
plt.hist(gray_img.ravel(), 256, [0, 256])
plt.show()
# To show an image in a notebook/exam environment
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```
**Tip 3:** Histogram = Contrast/Thresholding Tool and Subplots = Comparison Tool. Focus your space on the OpenCV functions like `cv2.findHomography` and `cv2.Canny`, as those are the "action" steps the robot actually needs to move.
