# src/core/feature_extraction.py
import numpy as np
import cv2

def extract_features(image_rgb, leaf_mask, lesion_mask, lesion_contours):
    """
    Ekstrak fitur berbasis threshold (sesuai PDF):
        1. median_hue (lesion)
        2. entropy (histogram Hue lesi)
        3. lesion_area_ratio
        4. num_lesions
        5. avg_circularity
    """
    features = {}

    # 1. Median Hue lesi
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue_channel = hsv[:, :, 0]  # Hue: 0-179 (OpenCV)
    lesion_pixels = hue_channel[lesion_mask > 0]
    features['median_hue'] = np.median(lesion_pixels) if len(lesion_pixels) > 0 else 0.0

    # 2. Entropy dari histogram Hue lesi (lebih ringan & robust daripada GLCM)
    if len(lesion_pixels) > 0:
        hist, _ = np.histogram(lesion_pixels, bins=32, range=(0, 180))
        hist = hist / (hist.sum() + 1e-6)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features['entropy'] = float(entropy)
    else:
        features['entropy'] = 0.0

    # 3. Rasio luas lesi / daun
    leaf_area = np.sum(leaf_mask > 0)
    lesion_area = np.sum(lesion_mask > 0)
    features['lesion_area_ratio'] = lesion_area / leaf_area if leaf_area > 0 else 0.0

    # 4. Jumlah lesi
    features['num_lesions'] = len(lesion_contours)

    # 5. Rata-rata circularity
    circularities = []
    for c in lesion_contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(circularity)
    features['avg_circularity'] = np.mean(circularities) if circularities else 0.0

    return features