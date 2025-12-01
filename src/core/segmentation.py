# src/core/segmentation.py
import cv2
import numpy as np
from ..utils.helpers import overlay_contours

def is_mango_leaf(contour):
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False
    aspect_ratio = w / h
    return 0.2 < aspect_ratio < 0.7

def segment_leaf(image_rgb, h_min=35, h_max=85, s_min=50, v_min=20, min_area_ratio=0.01):
    h_img, w_img = image_rgb.shape[:2]
    min_area = max(int(h_img * w_img * min_area_ratio), 800)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        leaf_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        overlay = image_rgb.copy()
        return leaf_mask, None, overlay

    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area and is_mango_leaf(c)]
    leaf_contour = max(valid_contours or contours, key=cv2.contourArea)

    leaf_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(leaf_mask, [leaf_contour], -1, 255, thickness=cv2.FILLED)
    
    # Garis hijau tebal (ketebalan 3)
    overlay = image_rgb.copy()
    overlay = overlay_contours(overlay, [leaf_contour], color=(0, 255, 0), thickness=3)
    
    return leaf_mask, leaf_contour, overlay