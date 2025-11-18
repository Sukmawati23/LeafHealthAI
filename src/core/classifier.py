# src/core/classifier.py

def classify_condition(features):
    """
    Klasifikasi berbasis aturan (sesuai PDF).
    Returns: (label, confidence, recommendation)
    """
    hue = features['median_hue']
    entropy = features['entropy']
    area_ratio = features['lesion_area_ratio']
    circularity = features['avg_circularity']
    num_lesions = features['num_lesions']

    # Sehat: lesi sangat kecil atau tidak ada
    if area_ratio < 0.02 or num_lesions == 0:
        return "🌱 Sehat", 1.0, "Daun dalam kondisi baik. Pertahankan pemeliharaan rutin."

    # Jamur: lesi coklat/kuning (hue rendah), entropy tinggi, area besar
    if entropy > 0.7 and hue < 35 and area_ratio > 0.05:
        return "🍄 Jamur", 0.85, "Gunakan fungisida berbasis tembaga atau chlorothalonil. Pangkas daun sakit."

    # Bakteri: lesi basah, tepi kuning (hue 40-70), circularity rendah
    if 40 < hue < 70 and circularity < 0.6 and area_ratio > 0.03:
        return "🦠 Bakteri", 0.80, "Hindari siram daun. Gunakan bakterisida (misal: streptomycin). Tingkatkan sirkulasi udara."

    # Default: Hama atau defisiensi nutrisi
    return "🐛 Hama/Defisiensi", 0.75, "Periksa adanya serangga. Berikan pupuk NPK seimbang atau kalsium jika ujung daun mengering."