# src/utils/helpers.py
import cv2
import numpy as np
from PIL import Image

def load_image(filepath):
    """Load image using OpenCV and convert BGR → RGB."""
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        raise ValueError(f"Gambar tidak ditemukan atau format tidak didukung: {filepath}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def save_image(filepath, img_rgb):
    """Save RGB image to file as BGR."""
    if img_rgb.dtype != np.uint8:
        img_rgb = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)

def overlay_contours(image, contours, color=(0, 255, 0), thickness=2):
    """Draw contours on image copy (RGB format)."""
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay

def compute_circularity(contour):
    """Calculate circularity = 4π·area / perimeter²."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)

def is_mango_leaf(contour):
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False
    aspect_ratio = w / h
    # Daun mangga: umumnya lebih panjang → aspect ratio < 1 (tinggi > lebar)
    return 0.2 < aspect_ratio < 0.7  # Sesuaikan setelah uji coba