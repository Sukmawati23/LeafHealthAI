# src/gui/main_window.py
import tkinter as tk
import os
import tempfile
import threading
from tkinter import filedialog, messagebox
from datetime import datetime
from PIL import Image, ImageDraw, ImageTk
import cv2
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from ..core.segmentation import segment_leaf
from ..core.lesion_detection import detect_lesions
from ..core.feature_extraction import extract_features
from ..core.classifier import classify_condition
from ..utils.helpers import load_image, save_image


class LeafHealthAIApp:
    def __init__(self):
        self.root = ttk.Window(themename="morph")
        self.root.title("LeafHealthAI — Deteksi Penyakit Daun Mangga (Jamur/Bakteri)")
        self.root.geometry("1400x820")
        self.root.minsize(1200, 700)

        try:
            icon = self._create_icon()
            if icon:
                self.root.iconphoto(True, icon)
        except Exception as e:
            print("ℹ️ Icon tidak dimuat (aman diabaikan):", e)

        self.original_rgb = None
        self.resized_rgb = None
        self.overlay_img = None
        self.crop_lesion = None
        self.features = None
        self.prediction = None
        self.is_processing = False

        self.setup_ui()

    def _create_icon(self):
        img = Image.new('RGBA', (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((4, 4, 28, 28), fill=(46, 184, 46, 255))
        draw.polygon([(16, 8), (24, 16), (16, 24)], fill=(30, 120, 30, 255))
        return ImageTk.PhotoImage(img)

    def setup_ui(self):
        # Navbar
        navbar = ttk.Frame(self.root, bootstyle="dark", padding=5)
        navbar.pack(fill="x")
        ttk.Label(navbar, text="LeafHealthAI", font=("Segoe UI", 16, "bold"), foreground="white").pack(side="left", padx=10)
        ttk.Label(navbar, text="Deteksi Jamur/Bakteri pada Daun Mangga", font=("Segoe UI", 10), foreground="lightgray").pack(side="left", padx=(0, 20))
        ttk.Button(navbar, text="ℹ️ Panduan", bootstyle="light-outline", command=self.show_help).pack(side="right", padx=5)

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Control panel
        control_frame = ttk.Labelframe(main_frame, text="Parameter & Kontrol", padding=10)
        control_frame.pack(side="left", fill="y", expand=False, padx=(0, 10))

        self.upload_btn = ttk.Button(control_frame, text="Upload Citra Daun", bootstyle="success-outline", command=self.upload_image, width=25)
        self.upload_btn.pack(pady=(0, 15), ipady=5)

        seg_frame = ttk.Labelframe(control_frame, text="Segmentasi Daun (HSV)", padding=8)
        seg_frame.pack(fill="x", pady=(0, 15))
        self.h_min_var = tk.IntVar(value=35)
        self.h_max_var = tk.IntVar(value=85)
        self.s_min_var = tk.IntVar(value=50)
        self._add_slider(seg_frame, "Hue Min", self.h_min_var, 0, 179)
        self._add_slider(seg_frame, "Hue Max", self.h_max_var, 0, 179)
        self._add_slider(seg_frame, "Sat Min", self.s_min_var, 0, 255)

        lesion_frame = ttk.Labelframe(control_frame, text="Deteksi Lesi (Hue)", padding=8)
        lesion_frame.pack(fill="x", pady=(0, 15))
        self.hue_min_var = tk.IntVar(value=0)
        self.hue_max_var = tk.IntVar(value=40)
        self._add_slider(lesion_frame, "Hue Min", self.hue_min_var, 0, 179)
        self._add_slider(lesion_frame, "Hue Max", self.hue_max_var, 0, 179)

        self.analyze_btn = ttk.Button(control_frame, text="Analisis Sekarang", bootstyle="primary", command=self.start_analysis, width=25)
        self.analyze_btn.pack(pady=10, ipady=5)

        self.save_btn = ttk.Button(control_frame, text="Simpan Laporan (PDF)", bootstyle="warning-outline", command=self.save_report, state="disabled")
        self.save_btn.pack(pady=5, ipady=3)

        # 6-panel visualisasi (Original, Resize, Segmentasi, Deteksi Lesi, Crop, Hasil)
        viz_frame = ttk.Labelframe(main_frame, text="Tahapan Praproses & Visualisasi", padding=5)
        viz_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        stages_frame = ttk.Frame(viz_frame)
        stages_frame.pack(fill="both", expand=True)

        stage_labels = [
            "📸 Original",
            "📏 Resize",
            "🌿 Segmentasi Daun",
            "🔴 Deteksi Lesi",
            "🎯 Fokus Lesi (Crop)",
            "✅ Hasil Diagnosa"
        ]
        self.stage_canvases = []
        self.stage_labels = []

        # create a 2x3 grid for 6 panels
        for i, title in enumerate(stage_labels):
            col = ttk.Labelframe(stages_frame, text=title, padding=3)
            r = i // 3
            c = i % 3
            col.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
            stages_frame.columnconfigure(c, weight=1)
            stages_frame.rowconfigure(r, weight=1)

            canvas = tk.Canvas(col, bg="#fafafa", width=320, height=240, highlightthickness=1, highlightbackground="#ddd")
            canvas.pack(fill="both", expand=True)
            self.stage_canvases.append(canvas)

            lbl = ttk.Label(col, text="Belum diproses", font=("Segoe UI", 8, "italic"), foreground="#777")
            lbl.pack(pady=1)
            self.stage_labels.append(lbl)

        # Hasil diagnosa (panel kanan)
        result_frame = ttk.Labelframe(main_frame, text="Hasil Diagnosa", padding=10)
        result_frame.pack(side="left", fill="both", expand=True)

        self.pred_card = ttk.Frame(result_frame, bootstyle="light", padding=10)
        self.pred_card.pack(fill="x", pady=(0, 15))
        self.pred_label = ttk.Label(self.pred_card, text="Menunggu analisis...", font=("Segoe UI", 14, "bold"), bootstyle="secondary")
        self.pred_label.pack()
        self.conf_label = ttk.Label(self.pred_card, text="", font=("Segoe UI", 10))
        self.conf_label.pack()

        ttk.Label(result_frame, text="Rekomendasi:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.recom_text = tk.Text(result_frame, height=4, wrap="word", font=("Segoe UI", 10), bg=self.root.cget("background"), relief="flat", state="disabled")
        self.recom_text.pack(fill="x", pady=(5, 15))

        self.feat_frame = ttk.Labelframe(result_frame, text="Fitur yang Diekstraksi", padding=8)
        self.feat_frame.pack(fill="both", expand=True)
        self.feat_tree = ttk.Treeview(self.feat_frame, columns=("value",), show="tree headings", height=8)
        self.feat_tree.heading("#0", text="Fitur")
        self.feat_tree.heading("value", text="Nilai")
        self.feat_tree.column("#0", width=180, anchor="w")
        self.feat_tree.column("value", width=90, anchor="e")
        self.feat_tree.pack(fill="both", expand=True, pady=2)

        self.status_var = tk.StringVar(value="Siap. Upload citra daun untuk mulai.")
        ttk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9), bootstyle="secondary", relief="sunken", anchor="w").pack(side="bottom", fill="x")

    def _add_slider(self, parent, label, var, min_val, max_val):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text=label, font=("Segoe UI", 9)).pack(anchor="w")
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient="horizontal")
        slider.pack(fill="x", pady=(2, 0))
        slider.bind("<ButtonRelease-1>", lambda e: var.set(int(var.get())))
        ttk.Label(frame, textvariable=var, font=("Courier", 8), foreground="#666").pack(anchor="e")

    def display_stage_image(self, stage_idx, img_rgb):
        """
        Menampilkan gambar pada canvas stage_idx.
        img_rgb diasumsikan dalam format RGB uint8.
        """
        canvas = self.stage_canvases[stage_idx]
        h, w = img_rgb.shape[:2]
        cw, ch = int(canvas.winfo_width() or 320), int(canvas.winfo_height() or 240)
        # jika ukuran canvas belum ter-render, gunakan default 320x240
        cw = cw if cw > 10 else 320
        ch = ch if ch > 10 else 240
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        tk_img = ImageTk.PhotoImage(Image.fromarray(img_resized))
        canvas.delete("all")
        x, y = (cw - nw) // 2, (ch - nh) // 2
        canvas.create_image(x, y, anchor="nw", image=tk_img)
        canvas.image = tk_img
        # update label kecil di bawah canvas
        self.stage_labels[stage_idx].config(text="✓ Siap")

    def upload_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath:
            return
        try:
            self.original_rgb = load_image(filepath)
            # reset beberapa state
            self.resized_rgb = None
            self.overlay_img = None
            self.crop_lesion = None
            self.features = None
            self.prediction = None

            self.display_stage_image(0, self.original_rgb)
            # set remaining panels to "Belum diproses"
            for i in range(1, 6):
                self.stage_labels[i].config(text="Belum diproses")
                self.stage_canvases[i].delete("all")
            self.pred_label.config(text="Citra dimuat", bootstyle="info")
            self.recom_text.config(state="normal")
            self.recom_text.delete(1.0, tk.END)
            self.recom_text.insert(tk.END, "Klik 'Analisis Sekarang' untuk memulai diagnosa.")
            self.recom_text.config(state="disabled")
            self.save_btn.config(state="disabled")
            self.status_var.set(f"✓ Citra dimuat: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat gambar:\n{str(e)}")

    def start_analysis(self):
        if self.original_rgb is None:
            messagebox.showwarning("Peringatan", "Harap upload citra terlebih dahulu!")
            return
        if self.is_processing:
            return
        self.analyze_btn.config(state="disabled", text="Memproses...")
        self.upload_btn.config(state="disabled")
        self.is_processing = True
        self.status_var.set("Sedang menganalisis citra...")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            # Tahap 0: Original (sudah ditampilkan di upload)
            self.root.after(0, lambda: self.display_stage_image(0, self.original_rgb))

            # Tahap 1: Resize (640x480)
            resized = cv2.resize(self.original_rgb, (640, 480), interpolation=cv2.INTER_AREA)
            self.resized_rgb = resized.copy()
            self.root.after(0, lambda img=resized: self.display_stage_image(1, img))

            # Tahap 2: Segmentasi (gunakan gambar resize)
            leaf_mask, leaf_contour, leaf_overlay = segment_leaf(
                self.resized_rgb,
                h_min=self.h_min_var.get(),
                h_max=self.h_max_var.get(),
                s_min=self.s_min_var.get()
            )
            # overlay leaf_overlay sudah RGB
            self.root.after(0, lambda img=leaf_overlay: self.display_stage_image(2, img))

            # Tahap 3: Deteksi Lesi (gunakan gambar resize)
            lesion_mask, lesion_contours, lesion_overlay = detect_lesions(
                self.resized_rgb,
                leaf_mask,
                hue_min=self.hue_min_var.get(),
                hue_max=self.hue_max_var.get()
            )
            self.root.after(0, lambda img=lesion_overlay: self.display_stage_image(3, img))

            # Tahap 4: Crop fokus lesi (otomatis ambil lesi terbesar)
            if lesion_contours and len(lesion_contours) > 0:
                main_cnt = max(lesion_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(main_cnt)
                # tambahkan margin sedikit agar tidak terpotong
                pad = int(0.15 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(self.resized_rgb.shape[1], x + w + pad)
                y2 = min(self.resized_rgb.shape[0], y + h + pad)
                crop = self.resized_rgb[y1:y2, x1:x2]
                # jika crop kosong karena ukuran kecil -> buat placeholder hitam
                if crop.size == 0:
                    crop = np.zeros((300, 300, 3), dtype=np.uint8)
                else:
                    # resize crop ke ukuran visual yang konsisten (300x300)
                    crop = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_AREA)
            else:
                crop = np.zeros((300, 300, 3), dtype=np.uint8)

            self.crop_lesion = crop.copy()
            self.root.after(0, lambda img=crop: self.display_stage_image(4, img))

            # Tahap 5: Ekstraksi fitur dan klasifikasi (gunakan mask & kontur dari resize)
            features = extract_features(self.resized_rgb, leaf_mask, lesion_mask, lesion_contours)
            label, conf, rec = classify_condition(features)

            # overlay final: gunakan lesion_overlay (dari detect_lesions)
            self.overlay_img = lesion_overlay
            self.features = features
            self.prediction = (label, conf, rec)

            # update UI akhir (panel hasil)
            self.root.after(0, self._update_ui_after_analysis)

        except Exception as e:
            self.root.after(0, lambda err=e: self._handle_analysis_error(err))
        finally:
            self.root.after(0, self._finish_processing)

    def _update_ui_after_analysis(self):
        label, conf, rec = self.prediction

        # Update panel ke-6 (index 5) dengan label ringkas
        try:
            self.stage_labels[5].config(text=f"{label}\n{conf:.0%}")
        except Exception:
            pass

        # Update panel kanan
        color_map = {"❓ Tidak Terdeteksi": "secondary", "🍄 Jamur": "warning", "🦠 Bakteri": "danger"}
        # gunakan first token (emoji) atau kata pertama sebagai key attempt
        key = label.split()[0] if isinstance(label, str) else label
        style = color_map.get(key, "secondary")
        self.pred_label.config(text=label, bootstyle=style)
        self.conf_label.config(text=f"Akurasi estimasi: {conf:.0%}")

        self.recom_text.config(state="normal")
        self.recom_text.delete(1.0, tk.END)
        self.recom_text.insert(tk.END, rec)
        self.recom_text.config(state="disabled")

        # tampilkan fitur di treeview
        for item in self.feat_tree.get_children():
            self.feat_tree.delete(item)
        if isinstance(self.features, dict):
            for k, v in self.features.items():
                if isinstance(v, float):
                    val_str = f"{v:.4f}"
                else:
                    val_str = str(v)
                self.feat_tree.insert("", "end", text=k.replace("_", " ").title(), values=(val_str,))

        # enable save report
        self.save_btn.config(state="normal")
        self.status_var.set("✓ Analisis selesai.")

    def _handle_analysis_error(self, e):
        messagebox.showerror("Error Analisis", f"Terjadi kesalahan:\n{str(e)}")
        self.pred_label.config(text="Gagal Analisis", bootstyle="danger")
        self.status_var.set("✗ Analisis gagal.")

    def _finish_processing(self):
        self.analyze_btn.config(state="normal", text="Analisis Sekarang")
        self.upload_btn.config(state="normal")
        self.is_processing = False

    def show_help(self):
        help_text = (
            "LeafHealthAI – Panduan\n\n"
            "1. Upload foto daun mangga (utuh atau close-up lesi).\n"
            "2. Klik [Analisis Sekarang].\n"
            "3. Sistem akan tampilkan tahapan:\n"
            "   • Original\n"
            "   • Resize ke 640×480\n"
            "   • Segmentasi daun (HSV)\n"
            "   • Deteksi lesi (lingkaran merah)\n"
            "   • Fokus lesi (crop otomatis)\n"
            "   • Hasil diagnosa (Jamur/Bakteri) beserta rekomendasi\n\n"
            "Catatan: Aplikasi ini hanya untuk daun mangga dan berbasis aturan sederhana."
        )
        messagebox.showinfo("Panduan", help_text)

    def save_report(self):
        # Sesuaikan dengan overlay_img dan self.prediction
        filepath = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not filepath:
            return
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
            if self.overlay_img is None:
                # fallback: simpan resized atau crop
                if self.resized_rgb is not None:
                    save_image(temp_path, self.resized_rgb)
                elif self.original_rgb is not None:
                    save_image(temp_path, self.original_rgb)
                else:
                    # buat placeholder
                    placeholder = np.zeros((300, 300, 3), dtype=np.uint8)
                    save_image(temp_path, placeholder)
            else:
                save_image(temp_path, self.overlay_img)

            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader

            c = canvas.Canvas(filepath, pagesize=A4)
            w, h = A4
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, h - 50, "🌿 LeafHealthAI — Laporan Diagnosa")
            c.setFont("Helvetica", 10)
            c.drawString(50, h - 70, f"Dibuat: {datetime.now().strftime('%d %B %Y, %H:%M')}")

            if os.path.exists(temp_path):
                c.drawImage(ImageReader(temp_path), 50, h - 360, width=500, height=280)

            y = h - 380
            label, conf, rec = self.prediction if self.prediction is not None else ("-", 0.0, "-")
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, f"Diagnosa: {label}")
            y -= 20
            c.setFont("Helvetica", 11)
            c.drawString(50, y, f"Akurasi Estimasi: {conf:.0%}")
            y -= 25
            c.drawString(50, y, "Rekomendasi:")
            y -= 16
            for line in rec.split(". "):
                if line.strip():
                    c.drawString(70, y, f"• {line.strip()}.")
                    y -= 14
            y -= 10
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y, "Fitur yang Digunakan:")
            y -= 18
            c.setFont("Helvetica", 9)
            if isinstance(self.features, dict):
                for k, v in self.features.items():
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    c.drawString(70, y, f"{k.replace('_', ' ').title():<22} : {val}")
                    y -= 14
                    if y < 100:
                        c.showPage()
                        y = h - 50
            c.save()
            os.unlink(temp_path)
            messagebox.showinfo("Berhasil", "Laporan PDF berhasil disimpan!")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan PDF:\n{str(e)}")

    def run(self):
        self.root.mainloop()


# if run as script (optional)
if __name__ == "__main__":
    app = LeafHealthAIApp()
    app.run()
