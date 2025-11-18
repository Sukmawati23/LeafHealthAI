# src/core/lesion_detection.py
import cv2
import numpy as np
from skimage import color
from ..utils.helpers import overlay_contours

def detect_lesions(image_rgb, leaf_mask, a_star_thresh=130, min_area=200):
    """
    Deteksi lesi menggunakan thresholding saluran a* (Lab).
    Returns: (lesion_mask, lesion_contours, overlay_with_red_lesions)
    """
    if leaf_mask is None or not leaf_mask.any():
        h, w = image_rgb.shape[:2]
        lesion_mask = np.zeros((h, w), dtype=np.uint8)
        return lesion_mask, [], image_rgb.copy()

    lab = color.rgb2lab(image_rgb)
    a_star = lab[:, :, 1]  # a*: hijau (-) → merah (+)

    _, lesion_thresh = cv2.threshold(a_star.astype(np.uint8), a_star_thresh, 255, cv2.THRESH_BINARY)

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