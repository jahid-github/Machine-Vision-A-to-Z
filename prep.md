This comprehensive study guide is designed to prepare you for your HAMK Robotics Machine Vision final exam, covering all five key lecture modules. It includes theoretical foundations, mathematical concepts, and hands-on Python/OpenCV code snippets.

---

### **Module 1: Introduction to Machine Vision & Tools**

Machine vision allows robots to perceive and interact with their environment by converting visual data into actionable information.

* **Image Representation:** Images are stored as 2D/3D matrices.
* **Grayscale:** A single 2D matrix where pixel values range from 0 (Black) to 255 (White).
* **Color (BGR):** A 3D matrix (height, width, 3 channels). Note that OpenCV uses **BGR** (Blue, Green, Red) by default, not RGB.
* **Coordinate Systems:**
* **Image Coordinates:** Origin (0,0) is the top-left corner. $u$ moves right, $v$ moves down.
* **RoboDK Integration:** Use `RDK.Cam2D_Snapshot(file, cam)` to capture images in simulation and `cv2.imread(file)` to load them for processing.
---

### **Module 2: Low-Level Vision I – Filtering & Enhancement**

Preprocessing is critical to remove noise and enhance features before analysis.

* **Point Operations:** Operations that change a pixel's value based only on its original value (e.g., contrast/brightness).
* **Histogram Equalization:** A technique to improve image contrast by spreading out the most frequent intensity values.
* **Spatial Filtering (Convolution):** A "kernel" (small matrix) slides over the image to perform mathematical operations.
* **Smoothing Filters:**
* **Mean/Average:** Blurs the image to reduce noise.
* **Gaussian:** Uses a weighted average; better for "Gaussian" (random) noise.
* **Median:** Replaces the center pixel with the median of neighbors; excellent for **Salt-and-Pepper noise** (black/white dots).
---
### **Module 3: Low-Level Vision II – Segmentation & Edges**

Segmentation converts a grayscale image into a binary mask (Black & White) to isolate objects.

* **Thresholding Methods:**
* **Global:** Uses a fixed value (e.g., all pixels > 127 become 255).
* **Otsu’s Binarization:** Automatically calculates the optimal threshold by minimizing intra-class variance.
* **Adaptive Thresholding:** Calculates thresholds for small regions; best for images with uneven lighting.
* **Edge Detection:**
* **Sobel:** Uses gradients to find intensity changes.
* **Canny:** A multi-stage, robust detector that is the industry standard for clean edges.
* **Morphological Operations:**
* **Erosion:** Shrinks white regions (removes small noise).
* **Dilation:** Expands white regions (closes small holes).
* **Opening:** Erosion followed by Dilation (removes noise while keeping object size).
* **Closing:** Dilation followed by Erosion (closes holes while keeping object size).
---
### **Module 4: Coordinate Mapping (Calibration)**
This links "Pixel World" to "Robot World".
* **Key Concept:** Cameras see in pixels ($u, v$), but robots move in millimeters ($X, Y, Z$).
* **2D Transformations:**
* **Translation:** Moving an object ($x+dx, y+dy$).
* **Rigid (Euclidean):** Rotation + Translation (3 Degrees of Freedom).
* **Affine:** Rotation + Translation + Scale + Shear (6 Degrees of Freedom).
* **Homography:** A projective transformation used for planar surfaces. Requires at least **4 pairs of points** to calculate the mapping matrix.
  
* **Calibration Workflow:**
1. Collect 4+ points in pixels (from camera) and their corresponding real-world coordinates (from robot).
2. Use `cv2.findHomography()` to find the transformation matrix.
3. Use `cv2.perspectiveTransform()` to convert new pixel detections to robot coordinates.
---

### **Module 5: Color & Shape Analysis**

Combining descriptors to uniquely identify objects.

* **HSV Color Space:** Robust against lighting changes.
* **Hue (H):** The color (0-179 in OpenCV).
* **Saturation (S):** Vibrancy.
* **Value (V):** Brightness.
* **Geometric Descriptors:**
* **Aspect Ratio:** Width / Height.
* **Circularity:** $4\pi \times (\text{Area} / \text{Perimeter}^2)$. A perfect circle has a circularity of 1.0.
* **SimpleBlobDetector:** An OpenCV tool that automates detection based on area, circularity, convexity, and inertia.
---
### **Final Exam Quick-Reference Code**

```python
import cv2
import numpy as np

# 1. PREPROCESSING (Module 2)
img = cv2.imread('snapshot.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. SEGMENTATION (Module 3)
_, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((5,5), np.uint8)
cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 3. COLOR DETECTION (Module 5)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
red_mask = cv2.inRange(hsv, lower_red, upper_red)

# 4. COORDINATE MAPPING (Module 4)
# pixels: [[u1,v1], [u2,v2]...] | robot: [[x1,y1], [x2,y2]...]
H, _ = cv2.findHomography(src_pixels, dst_robot)
target_pixel = np.array([[[u, v]]], dtype='float32')
robot_coord = cv2.perspectiveTransform(target_pixel, H)

```
**Pro-Tip for the Exam:** If a question asks why we use HSV instead of BGR, the answer is that **HSV separates color (Hue) from lighting (Value)**, making it much more stable in industrial environments.
