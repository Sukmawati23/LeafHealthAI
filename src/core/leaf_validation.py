import cv2
import numpy as np

def validate_leaf_image(image_path):
    """
    Validasi sederhana: cek apakah gambar mengandung area hijau dominan.
    Tidak pakai model AI.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Rentang hijau: Hue 35-85, Saturation > 50
    green_mask = cv2.inRange(hsv, (35, 50, 30), (85, 255, 255))
    green_ratio = cv2.countNonZero(green_mask) / green_mask.size
    return green_ratio > 0.1  # Minimal 10% hijau