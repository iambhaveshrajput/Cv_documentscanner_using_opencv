# 📄 OpenCV Mobile Document Scanner

This project implements a mobile-style document scanner using Computer Vision techniques in Python. It automatically detects the four corners of a document in a photo, applies a **perspective transformation** to correct for any angle or distortion, and enhances the result to a clean, black-and-white scan.

The core logic is demonstrated using OpenCV and other common computer vision libraries, originally developed in a Google Colab environment.

---

## ✨ Features

* **Automatic Edge Detection:** Uses the Canny algorithm to find precise document boundaries.
* **Contour Finding:** Locates the largest four-sided contour (the document) in the image.
* **Perspective Correction:** Applies a geometric transformation to simulate a flat, overhead scan, correcting for keystoning and perspective distortion.
* **Image Enhancement:** Uses **adaptive thresholding** (`skimage.filters.threshold_local`) to produce a crisp, high-contrast B/W "scanned" image.

---

## ⚙️ Technical Details (Processing Pipeline)

The document scanning process follows these steps:

1.  **Detection:** Preprocess the image (grayscale, blur), run **Canny edge detection**, and identify the largest 4-point contour.
2.  **Transformation:** Calculate the perspective matrix and use `cv2.warpPerspective` to flatten the image.
3.  **Enhancement:** Apply **adaptive Gaussian thresholding** on the grayscale, warped image to produce the final B/W scan.

---

## 🛠️ Requirements

To run the provided code, you need the following libraries:

```bash
pip install opencv-python imutils scikit-image numpy matplotlib
