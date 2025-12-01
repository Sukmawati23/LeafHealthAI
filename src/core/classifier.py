# src/core/classifier.py
def classify_condition(features):
    """
    Klasifikasi khusus daun mangga: hanya Jamur vs Bakteri.
    Sesuai dokumen PDF.
    """
    hue = features['median_hue']
    entropy = features['entropy']
    circularity = features['avg_circularity']
    area_ratio = features['lesion_area_ratio']
    num_lesions = features['num_lesions']

    if num_lesions == 0 or area_ratio < 0.01:
        return "❓ Tidak Terdeteksi", 0.5, "Tidak ditemukan lesi signifikan. Pastikan foto menangkap bercak penyakit."

    # Jamur: lesi gelap (Hue < 30), tekstur kasar (entropi > 0.3), bentuk tidak beraturan (circularity < 0.5)
    if hue < 30 and entropy > 0.3 and circularity < 0.5:
        return "🍄 Jamur", 0.9, "Kemungkinan besar penyakit jamur (anthracnose). Semprot dengan fungisida berbasis tembaga. Hindari kelembapan tinggi."

    # Bakteri: lesi cokelat kemerahan (Hue 30–50), tepi buram → circularity sangat rendah
    else:
        return "🦠 Bakteri", 0.85, "Kemungkinan infeksi bakteri (bacterial black spot). Hindari menyiram daun di sore hari. Gunakan tembaga + streptomycin jika tersedia."