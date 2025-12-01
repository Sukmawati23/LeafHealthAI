# src/core/lesion_detection.py
import cv2
import numpy as np

def detect_lesions(image_rgb, leaf_mask, hue_min=0, hue_max=40, min_area=150):
    if leaf_mask is None or not leaf_mask.any():
        h, w = image_rgb.shape[:2]
        lesion_mask = np.zeros((h, w), dtype=np.uint8)
        return lesion_mask, [], image_rgb.copy()

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue_channel = hsv[:, :, 0]
    lesion_thresh = cv2.inRange(hue_channel, hue_min, hue_max)
    lesion_thresh = cv2.bitwise_and(lesion_thresh, lesion_thresh, mask=leaf_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lesion_mask = cv2.morphologyEx(lesion_thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Gambar LINGKARAN MERAH di sekitar setiap lesi
    overlay = image_rgb.copy()
    for cnt in lesion_contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = max(int(radius), 5)
        cv2.circle(overlay, center, radius, (255, 0, 0), 2)  # Lingkaran merah, ketebalan 2

    return lesion_mask, lesion_contours, overlay