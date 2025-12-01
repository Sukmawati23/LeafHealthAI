# src/core/classifier.py
def classify_condition(features):
    """
    Klasifikasi KHUSUS daun mangga: hanya Bakteri vs Sehat.
    Returns: (label, confidence, recommendation)
    """
    hue = features['median_hue']
    entropy = features['entropy']
    area_ratio = features['lesion_area_ratio']
    circularity = features['avg_circularity']
    num_lesions = features['num_lesions']

    # Jika TIDAK ADA lesi → anggap sehat
    if num_lesions == 0 or area_ratio < 0.01:
        return "🌱 Sehat", 0.95, "Daun mangga tampak sehat. Pertahankan perawatan."

    # Bakteri (Bacterial Black Spot): bercak hitam/basah, tepi kuning, circularity rendah
    if hue < 40 and circularity < 0.6 and area_ratio > 0.015:
        return "🦠 Bakteri (Black Spot)", 0.85, "Kemungkinan Bacterial Black Spot. Hindari siram daun. Gunakan tembaga + streptomycin jika tersedia. Tingkatkan sirkulasi udara."

    # Jika masih ada lesi tapi tidak memenuhi kriteria ketat → tetap anggap bakteri
    return "🦠 Bakteri (kemungkinan)", 0.75, "Lesi terdeteksi — cenderung bakteri pada daun mangga."