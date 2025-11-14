
import cv2
import numpy as np
import imutils
from skimage.filters import threshold_local
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
print("Upload an image of a document (photo):")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Load image
image = cv2.imread(image_path)
orig = image.copy()
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height=500)

# Step 1: Edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

plt.figure(figsize=(8,6))
plt.imshow(edged, cmap='gray')
plt.title("Step 1: Edges")
plt.axis("off")
plt.show()

# Step 2: Find contours and document outline
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    print("WARNING: Could not find contour of document!")
else:
    # show the outline
    outline = image.copy()
    cv2.drawContours(outline, [screenCnt], -1, (0,255,0), 2)
    outline_rgb = cv2.cvtColor(outline, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(outline_rgb)
    plt.title("Step 2: Document Outline")
    plt.axis("off")
    plt.show()

    # Step 3: Apply perspective transform
    pts = screenCnt.reshape(4, 2) * ratio
    
    def order_points(pts):
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # convert to grayscale + threshold for B/W effect
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped_gray, 11, offset=10, method="gaussian")
    warped_thresh = (warped_gray > T).astype("uint8") * 255

    plt.figure(figsize=(8,6))
    plt.imshow(warped_thresh, cmap='gray')
    plt.title("Step 3: Scanned Result")
    plt.axis("off")
    plt.show()
