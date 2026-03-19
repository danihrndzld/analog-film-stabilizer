import os
import sys
import glob
import threading
from pathlib import Path
import subprocess
import importlib.util

# --- Auto dependency check ---

def ensure_package(pkg_name, import_name=None):
    import_name = import_name or pkg_name
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

if not getattr(sys, "frozen", False):
    try:
        ensure_package("opencv-python", "cv2")
        ensure_package("numpy")
    except Exception as e:
        print("No pude instalar dependencias automáticamente:", e)

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Drag & drop support (optional)
DND_OK = False
try:
    if not getattr(sys, "frozen", False):
        ensure_package("tkinterdnd2")
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_OK = True
except Exception:
    DND_OK = False

VALID_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")


def list_images(folder):
    files = set()
    for ext in VALID_EXTS:
        files.update(glob.glob(os.path.join(folder, ext)))
        files.update(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(files)


def moving_average(points, radius=9):
    xs = np.array([p[0] if p is not None else np.nan for p in points], dtype=np.float32)
    ys = np.array([p[1] if p is not None else np.nan for p in points], dtype=np.float32)

    def fill_nans(arr):
        idx = np.arange(len(arr))
        good = np.isfinite(arr)
        if not np.any(good):
            return arr
        arr[~good] = np.interp(idx[~good], idx[good], arr[good])
        return arr

    xs = fill_nans(xs)
    ys = fill_nans(ys)

    k = radius * 2 + 1
    kernel = np.ones(k, dtype=np.float32) / k
    xs_s = np.convolve(np.pad(xs, (radius, radius), mode="edge"), kernel, mode="valid")
    ys_s = np.convolve(np.pad(ys, (radius, radius), mode="edge"), kernel, mode="valid")
    return list(zip(xs_s.tolist(), ys_s.tolist()))


def detect_perforation(frame, roi_ratio=0.22, threshold=210):
    h, w = frame.shape[:2]
    roi_w = max(50, int(w * roi_ratio))
    roi = frame[:, :roi_w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1
    best_cnt = None
    best_box = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh + 1e-6)
        if not (0.40 <= aspect <= 1.20):
            continue

        fill_ratio = area / float((bw * bh) + 1e-6)
        if fill_ratio < 0.75:
            continue

        if x > roi_w * 0.70:
            continue

        score = area + (fill_ratio * 1000.0)
        if score > best_score:
            best_score = score
            best_cnt = cnt
            best_box = (x, y, bw, bh)

    if best_cnt is None:
        return None

    x, y, bw, bh = best_box
    M = cv2.moments(best_cnt)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = x + bw / 2.0
        cy = y + bh / 2.0

    return (float(cx), float(cy))


def stabilize_folder(input_dir, output_dir, progress_cb=None, log_cb=None, roi_ratio=0.22, threshold=210, smooth_radius=9):
    files = list_images(input_dir)
    if not files:
        raise RuntimeError("No encontré imágenes dentro de la carpeta.")

    os.makedirs(output_dir, exist_ok=True)

    points = []
    failures = 0
    total = len(files)

    def log(msg):
        if log_cb:
            log_cb(msg)

    log(f"Encontré {total} imágenes.")
    log("Primera pasada: detectando perforación...")

    for i, f in enumerate(files, 1):
        frame = cv2.imread(f)
        if frame is None:
            points.append(None)
            failures += 1
            log(f"No pude abrir: {os.path.basename(f)}")
        else:
            pt = detect_perforation(frame, roi_ratio=roi_ratio, threshold=threshold)
            points.append(pt)
            if pt is None:
                failures += 1
                log(f"Sin detección: {os.path.basename(f)}")
        if progress_cb:
            progress_cb(i / (total * 2))

    valid = [p for p in points if p is not None]
    if not valid:
        raise RuntimeError("No logré detectar la perforación en ningún frame.")

    # Compute robust target from all valid detections (ignore smoothing for the anchor)
    target_x = float(np.median([p[0] for p in valid]))
    target_y = float(np.median([p[1] for p in valid]))
    log(f"Punto fijo objetivo: x={target_x:.2f}, y={target_y:.2f}")

    # Reject outliers: detections farther than 3×MAD from the median are treated as missed
    mad_x = float(np.median([abs(p[0] - target_x) for p in valid])) or 1.0
    mad_y = float(np.median([abs(p[1] - target_y) for p in valid])) or 1.0
    outlier_thresh_x = max(3.0 * mad_x, 30.0)
    outlier_thresh_y = max(3.0 * mad_y, 30.0)
    cleaned = [
        p if (p is not None
              and abs(p[0] - target_x) <= outlier_thresh_x
              and abs(p[1] - target_y) <= outlier_thresh_y)
        else None
        for p in points
    ]
    n_outliers = sum(1 for o, p in zip(cleaned, points) if o is None and p is not None)
    if n_outliers:
        log(f"Outliers descartados: {n_outliers}")

    # Fill None/outlier positions by linear interpolation (no smoothing — track real jitter)
    xs = np.array([p[0] if p is not None else np.nan for p in cleaned], dtype=np.float32)
    ys = np.array([p[1] if p is not None else np.nan for p in cleaned], dtype=np.float32)
    idx = np.arange(len(xs))
    good_x = np.isfinite(xs); good_y = np.isfinite(ys)
    if np.any(good_x): xs[~good_x] = np.interp(idx[~good_x], idx[good_x], xs[good_x])
    if np.any(good_y): ys[~good_y] = np.interp(idx[~good_y], idx[good_y], ys[good_y])
    per_frame = list(zip(xs.tolist(), ys.tolist()))

    shifts = [(target_x - pt[0], target_y - pt[1]) for pt in per_frame]
    crop_left   = int(np.ceil(max(max(dx, 0) for dx, dy in shifts)))
    crop_right  = int(np.ceil(max(max(-dx, 0) for dx, dy in shifts)))
    crop_top    = int(np.ceil(max(max(dy, 0) for dx, dy in shifts)))
    crop_bottom = int(np.ceil(max(max(-dy, 0) for dx, dy in shifts)))
    log(f"Crop aplicado — L:{crop_left} R:{crop_right} T:{crop_top} B:{crop_bottom}")
    log("Segunda pasada: estabilizando y guardando...")

    out_w = out_h = None

    for i, f in enumerate(files, 1):
        frame = cv2.imread(f)
        if frame is None:
            log(f"No pude abrir en segunda pasada: {os.path.basename(f)}")
        else:
            h, w = frame.shape[:2]
            pt = per_frame[i - 1]
            dx = target_x - pt[0]
            dy = target_y - pt[1]

            M = np.float32([[1, 0, dx], [0, 1, dy]])
            stabilized = cv2.warpAffine(
                frame,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            stabilized = stabilized[crop_top:h - crop_bottom, crop_left:w - crop_right]
            if out_h is None:
                out_h, out_w = stabilized.shape[:2]

            out_path = os.path.join(output_dir, os.path.basename(f))
            cv2.imwrite(out_path, stabilized)
        if progress_cb:
            progress_cb((total + i) / (total * 2))

    summary = {
        "total_frames": total,
        "failed_detections": failures,
        "target_x": round(target_x, 3),
        "target_y": round(target_y, 3),
        "output_width": out_w,
        "output_height": out_h,
        "applied_crop_left": crop_left,
        "applied_crop_right": crop_right,
        "applied_crop_top": crop_top,
        "applied_crop_bottom": crop_bottom,
    }

    with open(os.path.join(output_dir, "stabilization_report.txt"), "w", encoding="utf-8") as f:
        f.write("PERFORATION STABILIZATION REPORT\n")
        f.write("================================\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    log("Listo.")
    log(f"Frames: {summary['total_frames']}")
    log(f"Sin detección: {summary['failed_detections']}")
    log(f"Tamaño de salida: {out_w}×{out_h} px")
    return summary


class AppBase:
    def parse_drop_path(self, data):
        data = data.strip()
        if data.startswith("{") and data.endswith("}"):
            data = data[1:-1]
        return data

    def choose_input(self):
        folder = filedialog.askdirectory(title="Selecciona la carpeta de frames")
        if folder:
            self.set_input(folder)

    def choose_output(self):
        folder = filedialog.askdirectory(title="Selecciona la carpeta de salida")
        if folder:
            self.output_var.set(folder)

    def set_input(self, folder):
        self.input_var.set(folder)
        auto_out = str(Path(folder).parent / f"{Path(folder).name}_ESTABILIZADO")
        self.output_var.set(auto_out)
        self.log(f"Carpeta cargada: {folder}")

    def log(self, msg):
        self.root.after(0, lambda m=msg: (
            self.log_box.insert("end", m + "\n"),
            self.log_box.see("end"),
        ))

    def set_progress(self, value):
        self.root.after(0, lambda v=value: self.progress_var.set(max(0.0, min(100.0, v * 100.0))))

    def _finish(self, summary, output_dir):
        self.log("\nProceso terminado correctamente.")
        self.run_btn.config(state="normal")
        self.set_progress(1.0)
        messagebox.showinfo(
            "Listo",
            f"Secuencia estabilizada.\n\nSalida:\n{output_dir}\n\nSin detección: {summary['failed_detections']}",
        )

    def _error(self, e):
        self.log(f"ERROR: {e}")
        self.run_btn.config(state="normal")
        self.set_progress(1.0)
        messagebox.showerror("Error", str(e))

    def run_process(self):
        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Selecciona o arrastra una carpeta válida.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Falta la carpeta de salida.")
            return

        self.run_btn.config(state="disabled")
        self.progress_var.set(0)

        def worker():
            try:
                summary = stabilize_folder(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    progress_cb=self.set_progress,
                    log_cb=self.log,
                    roi_ratio=float(self.roi_var.get()),
                    threshold=int(self.threshold_var.get()),
                    smooth_radius=int(self.smooth_var.get()),
                )
                self.root.after(0, lambda: self._finish(summary, output_dir))
            except Exception as e:
                self.root.after(0, lambda err=e: self._error(err))

        threading.Thread(target=worker, daemon=True).start()


class DragDropApp(AppBase):
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.build_ui()

    def build_ui(self):
        self.root.title("Perforation Stabilizer")
        self.root.geometry("760x580")
        self.root.configure(padx=18, pady=18)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.progress_var = tk.DoubleVar(value=0)
        self.roi_var = tk.StringVar(value="0.22")
        self.threshold_var = tk.StringVar(value="210")
        self.smooth_var = tk.StringVar(value="9")

        tk.Label(self.root, text="Arrastra aquí la carpeta de frames", font=("Arial", 16, "bold")).pack(anchor="w")
        drop = tk.Label(self.root, text="⬇️ Suelta aquí la carpeta ⬇️", relief="groove", bd=2, height=5, bg="#f6f6f6")
        drop.pack(fill="x", pady=(8, 14))
        drop.drop_target_register(DND_FILES)
        drop.dnd_bind("<<Drop>>", self.on_drop)

        row1 = tk.Frame(self.root)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Input:", width=8, anchor="w").pack(side="left")
        tk.Entry(row1, textvariable=self.input_var).pack(side="left", fill="x", expand=True)
        tk.Button(row1, text="Elegir", command=self.choose_input).pack(side="left", padx=(8, 0))

        row2 = tk.Frame(self.root)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Output:", width=8, anchor="w").pack(side="left")
        tk.Entry(row2, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        tk.Button(row2, text="Elegir", command=self.choose_output).pack(side="left", padx=(8, 0))

        opts = tk.Frame(self.root)
        opts.pack(fill="x", pady=(10, 10))
        tk.Label(opts, text="ROI izq.").grid(row=0, column=0, sticky="w")
        tk.Entry(opts, textvariable=self.roi_var, width=8).grid(row=0, column=1, padx=(6, 16))
        tk.Label(opts, text="Threshold").grid(row=0, column=2, sticky="w")
        tk.Entry(opts, textvariable=self.threshold_var, width=8).grid(row=0, column=3, padx=(6, 16))
        tk.Label(opts, text="Suavizado").grid(row=0, column=4, sticky="w")
        tk.Entry(opts, textvariable=self.smooth_var, width=8).grid(row=0, column=5, padx=(6, 16))

        self.run_btn = tk.Button(self.root, text="Estabilizar secuencia", font=("Arial", 13, "bold"), command=self.run_process)
        self.run_btn.pack(fill="x", pady=(4, 10))

        ttk.Progressbar(self.root, variable=self.progress_var, maximum=100).pack(fill="x", pady=(0, 10))

        self.log_box = tk.Text(self.root, height=18, wrap="word")
        self.log_box.pack(fill="both", expand=True)
        self.log("Listo. Puedes arrastrar una carpeta o usar el botón Elegir.")

    def on_drop(self, event):
        folder = self.parse_drop_path(event.data)
        if os.path.isdir(folder):
            self.set_input(folder)
        else:
            messagebox.showerror("Error", "Lo que arrastraste no es una carpeta válida.")


class PickerApp(AppBase):
    def __init__(self):
        self.root = tk.Tk()
        self.build_ui()

    def build_ui(self):
        self.root.title("Perforation Stabilizer")
        self.root.geometry("760x560")
        self.root.configure(padx=18, pady=18)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.progress_var = tk.DoubleVar(value=0)
        self.roi_var = tk.StringVar(value="0.22")
        self.threshold_var = tk.StringVar(value="210")
        self.smooth_var = tk.StringVar(value="9")

        tk.Label(self.root, text="Perforation Stabilizer", font=("Arial", 16, "bold")).pack(anchor="w")
        tk.Label(self.root, text="Elige la carpeta de frames y el programa fijará la perforación en toda la secuencia.").pack(anchor="w", pady=(4, 12))

        row1 = tk.Frame(self.root)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Input:", width=8, anchor="w").pack(side="left")
        tk.Entry(row1, textvariable=self.input_var).pack(side="left", fill="x", expand=True)
        tk.Button(row1, text="Elegir", command=self.choose_input).pack(side="left", padx=(8, 0))

        row2 = tk.Frame(self.root)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Output:", width=8, anchor="w").pack(side="left")
        tk.Entry(row2, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        tk.Button(row2, text="Elegir", command=self.choose_output).pack(side="left", padx=(8, 0))

        opts = tk.Frame(self.root)
        opts.pack(fill="x", pady=(10, 10))
        tk.Label(opts, text="ROI izq.").grid(row=0, column=0, sticky="w")
        tk.Entry(opts, textvariable=self.roi_var, width=8).grid(row=0, column=1, padx=(6, 16))
        tk.Label(opts, text="Threshold").grid(row=0, column=2, sticky="w")
        tk.Entry(opts, textvariable=self.threshold_var, width=8).grid(row=0, column=3, padx=(6, 16))
        tk.Label(opts, text="Suavizado").grid(row=0, column=4, sticky="w")
        tk.Entry(opts, textvariable=self.smooth_var, width=8).grid(row=0, column=5, padx=(6, 16))

        self.run_btn = tk.Button(self.root, text="Estabilizar secuencia", font=("Arial", 13, "bold"), command=self.run_process)
        self.run_btn.pack(fill="x", pady=(4, 10))

        ttk.Progressbar(self.root, variable=self.progress_var, maximum=100).pack(fill="x", pady=(0, 10))

        self.log_box = tk.Text(self.root, height=18, wrap="word")
        self.log_box.pack(fill="both", expand=True)
        self.log("Listo. Usa el botón Elegir para seleccionar tu carpeta.")


def main():
    if DND_OK:
        try:
            app = DragDropApp()
        except Exception:
            app = PickerApp()
    else:
        app = PickerApp()
    app.root.mainloop()


if __name__ == "__main__":
    main()
