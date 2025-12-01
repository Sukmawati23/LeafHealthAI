# src/core/lesion_detection.py
import cv2
import numpy as np
from ..utils.helpers import overlay_contours

def detect_lesions(image_rgb, leaf_mask, hue_min=0, hue_max=40, min_area=150):
    """
    Deteksi lesi menggunakan thresholding saluran Hue (HSV).
    Lesi bakteri = hitam/coklat → Hue rendah.
    Returns: (lesion_mask, lesion_contours, overlay_with_red_lesions)
    """
    if leaf_mask is None or not leaf_mask.any():
        h, w = image_rgb.shape[:2]
        lesion_mask = np.zeros((h, w), dtype=np.uint8)
        return lesion_mask, [], image_rgb.copy()

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue_channel = hsv[:, :, 0]  # Hue: 0-179 (OpenCV)

    # Buat mask lesi berdasarkan Hue (lesion: Hue rendah = hitam/coklat)
    lesion_thresh = cv2.inRange(hue_channel, hue_min, hue_max)

    # Batasi hanya di area daun
    lesion_thresh = cv2.bitwise_and(lesion_thresh, lesion_thresh, mask=leaf_mask)

    # Kurangi noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lesion_mask = cv2.morphologyEx(lesion_thresh, cv2.MORPH_OPEN, kernel)

    # Cari kontur lesi
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Overlay kontur MERAH (RGB)
    overlay = image_rgb.copy()
    overlay = overlay_contours(overlay, lesion_contours, color=(255, 0, 0), thickness=2)

    return lesion_mask, lesion_contours, overlay