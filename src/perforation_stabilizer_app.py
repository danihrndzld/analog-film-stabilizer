import os
import glob

import cv2
import numpy as np

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


def _best_contour(thresh, roi_w, top_n=1):
    """Return top_n perforation candidates from a binary image as (cx, cy) tuples.

    When top_n=1 returns a single (cx, cy) tuple or None (backward-compatible).
    When top_n>1 returns a list of (cx, cy) tuples (may be shorter than top_n).
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
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
        candidates.append((score, cnt, (x, y, bw, bh)))

    candidates.sort(key=lambda c: c[0], reverse=True)
    results = []
    for score, cnt, (x, y, bw, bh) in candidates[:top_n]:
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + bw / 2.0
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else y + bh / 2.0
        results.append((float(cx), float(cy)))

    if top_n == 1:
        return results[0] if results else None
    return results


def detect_perforation(frame, roi_ratio=0.22, threshold=210, film_format='super8'):
    """Detect the perforation anchor point in a frame.

    film_format values:
      'super8'  — 1 perf, left ROI (default)
      '8mm'     — 2 perfs per frame, right ROI; midpoint used as anchor
      'super16' — 1 perf, right ROI
    """
    h, w = frame.shape[:2]
    roi_w = max(50, int(w * roi_ratio))

    roi_side = 'right' if film_format in ('8mm', 'super16') else 'left'
    roi_x_offset = (w - roi_w) if roi_side == 'right' else 0
    roi = frame[:, w - roi_w:] if roi_side == 'right' else frame[:, :roi_w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)

    def resolve(thresh_img):
        top_n = 2 if film_format == '8mm' else 1
        result = _best_contour(thresh_img, roi_w, top_n=top_n)
        if top_n == 1:
            if result is None:
                return None
            cx, cy = result
        else:
            if not result:
                return None
            if len(result) == 2:
                cx = (result[0][0] + result[1][0]) / 2.0
                cy = (result[0][1] + result[1][1]) / 2.0
            else:
                cx, cy = result[0]
        return (float(cx + roi_x_offset), float(cy))

    # Primary: global threshold
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    pt = resolve(thresh)
    if pt is not None:
        return pt

    # Fallback: adaptive threshold — handles overexposed frames where the
    # global threshold saturates the entire ROI.
    block = max(51, (min(h, roi_w) // 20) | 1)  # odd block size ~5% of ROI
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -5
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    return resolve(adaptive)


def stabilize_folder(input_dir, output_dir, progress_cb=None, log_cb=None,
                     roi_ratio=0.22, threshold=210, smooth_radius=9,
                     jpeg_quality=0, film_format='super8'):
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
            pt = detect_perforation(frame, roi_ratio=roi_ratio, threshold=threshold,
                                    film_format=film_format)
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

    # Fill None positions (failed detections) by linear interpolation.
    xs = np.array([p[0] if p is not None else np.nan for p in points], dtype=np.float32)
    ys = np.array([p[1] if p is not None else np.nan for p in points], dtype=np.float32)
    idx = np.arange(len(xs))
    good_x = np.isfinite(xs); good_y = np.isfinite(ys)
    if np.any(good_x): xs[~good_x] = np.interp(idx[~good_x], idx[good_x], xs[good_x])
    if np.any(good_y): ys[~good_y] = np.interp(idx[~good_y], idx[good_y], ys[good_y])
    per_frame = list(zip(xs.tolist(), ys.tolist()))

    log("Segunda pasada: estabilizando y guardando...")

    out_w = out_h = None

    for i, f in enumerate(files, 1):
        frame = cv2.imread(f)
        if frame is None:
            log(f"No pude abrir en segunda pasada: {os.path.basename(f)}")
        else:
            h, w = frame.shape[:2]
            if out_h is None:
                out_h, out_w = h, w
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

            basename = os.path.basename(f)
            if jpeg_quality == 0:
                basename = os.path.splitext(basename)[0] + ".png"
                out_path = os.path.join(output_dir, basename)
                cv2.imwrite(out_path, stabilized)
            else:
                out_path = os.path.join(output_dir, basename)
                cv2.imwrite(out_path, stabilized, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if progress_cb:
            progress_cb((total + i) / (total * 2))

    summary = {
        "total_frames": total,
        "failed_detections": failures,
        "target_x": round(target_x, 3),
        "target_y": round(target_y, 3),
        "output_width": out_w,
        "output_height": out_h,
        "output_format": "png (lossless)" if jpeg_quality == 0 else f"jpeg q{jpeg_quality}",
        "film_format": film_format,
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
