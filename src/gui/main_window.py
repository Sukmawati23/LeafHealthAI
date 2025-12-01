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
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)

        try:
            icon = self._create_icon()
            if icon:
                self.root.iconphoto(True, icon)
        except Exception as e:
            print("ℹ️ Icon tidak dimuat (aman diabaikan):", e)

        self.original_rgb = None
        self.overlay_img = None
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

        # 4-panel visualisasi
        viz_frame = ttk.Labelframe(main_frame, text="Tahapan Praproses & Visualisasi", padding=5)
        viz_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        stages_frame = ttk.Frame(viz_frame)
        stages_frame.pack(fill="both", expand=True)

        stage_labels = ["📸 Original", "🌿 Segmentasi Daun", "🔴 Deteksi Lesi", "✅ Hasil Diagnosa"]
        self.stage_canvases = []
        self.stage_labels = []

        for i, title in enumerate(stage_labels):
            col = ttk.Labelframe(stages_frame, text=title, padding=3)
            col.grid(row=0, column=i, padx=3, pady=3, sticky="nsew")
            stages_frame.columnconfigure(i, weight=1)

            canvas = tk.Canvas(col, bg="#fafafa", width=280, height=280, highlightthickness=1, highlightbackground="#ddd")
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
        self.feat_tree.column("#0", width=150, anchor="w")
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
        canvas = self.stage_canvases[stage_idx]
        h, w = img_rgb.shape[:2]
        cw, ch = 280, 280
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        tk_img = ImageTk.PhotoImage(Image.fromarray(img_resized))
        canvas.delete("all")
        x, y = (cw - nw) // 2, (ch - nh) // 2
        canvas.create_image(x, y, anchor="nw", image=tk_img)
        canvas.image = tk_img
        self.stage_labels[stage_idx].config(text="✓ Siap")

    def upload_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath:
            return
        try:
            self.original_rgb = load_image(filepath)
            self.display_stage_image(0, self.original_rgb)
            for i in range(1, 4):
                self.stage_labels[i].config(text="Belum diproses")
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
        self.analyze_btn.config(state="disabled", text="Memproses...")
        self.upload_btn.config(state="disabled")
        self.is_processing = True
        self.status_var.set("Sedang menganalisis citra...")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            # Tahap 1: Original
            self.root.after(0, lambda: self.display_stage_image(0, self.original_rgb))

            # Tahap 2: Segmentasi
            leaf_mask, _, leaf_overlay = segment_leaf(
                self.original_rgb,
                h_min=self.h_min_var.get(),
                h_max=self.h_max_var.get(),
                s_min=self.s_min_var.get()
            )
            self.root.after(0, lambda img=leaf_overlay: self.display_stage_image(1, img))

            # Tahap 3: Deteksi Lesi
            lesion_mask, lesion_contours, lesion_overlay = detect_lesions(
                self.original_rgb,
                leaf_mask,
                hue_min=self.hue_min_var.get(),
                hue_max=self.hue_max_var.get()
            )
            self.root.after(0, lambda img=lesion_overlay: self.display_stage_image(2, img))

            # Tahap 4: Klasifikasi
            features = extract_features(self.original_rgb, leaf_mask, lesion_mask, lesion_contours)
            label, conf, rec = classify_condition(features)

            self.overlay_img = lesion_overlay
            self.features = features
            self.prediction = (label, conf, rec)
            self.root.after(0, self._update_ui_after_analysis)

        except Exception as e:
            self.root.after(0, lambda err=e: self._handle_analysis_error(err))
        finally:
            self.root.after(0, self._finish_processing)

    def _update_ui_after_analysis(self):
        label, conf, rec = self.prediction

        # Update panel ke-4 (teks hasil)
        self.stage_labels[3].config(text=f"{label}\n{conf:.0%}")

        # Update panel kanan
        color_map = {"❓ Tidak Terdeteksi": "secondary", "🍄 Jamur": "warning", "🦠 Bakteri": "danger"}
        style = color_map.get(label.split()[0], "secondary")
        self.pred_label.config(text=label, bootstyle=style)
        self.conf_label.config(text=f"Akurasi estimasi: {conf:.0%}")

        self.recom_text.config(state="normal")
        self.recom_text.delete(1.0, tk.END)
        self.recom_text.insert(tk.END, rec)
        self.recom_text.config(state="disabled")

        for item in self.feat_tree.get_children():
            self.feat_tree.delete(item)
        for k, v in self.features.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            self.feat_tree.insert("", "end", text=k.replace("_", " ").title(), values=(val_str,))

        self.save_btn.config(state="normal")
        self.status_var.set("✓ Analisis selesai.")

    def _handle_analysis_error(self, e):
        messagebox.showerror("Error Analisis", f"Terjadi kesalahan:\n{str(e)}")
        self.pred_label.config(text="Gagal Analisis", bootstyle="danger")

    def _finish_processing(self):
        self.analyze_btn.config(state="normal", text="Analisis Sekarang")
        self.upload_btn.config(state="normal")
        self.is_processing = False

    def show_help(self):
        help_text = (
            "LeafHealthAI – Panduan\n\n"
            "1. Upload foto daun mangga (utuh atau close-up lesi).\n"
            "2. Klik [Analisis Sekarang].\n"
            "3. Sistem akan tampilkan:\n"
            "   • Garis hijau = tepi daun\n"
            "   • Lingkaran merah = lesi penyakit\n"
            "4. Hasil: Jamur atau Bakteri.\n\n"
            "Catatan: Aplikasi ini hanya untuk daun mangga."
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
            label, conf, rec = self.prediction
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