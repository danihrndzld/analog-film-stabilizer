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


def _best_contour(thresh, roi_w, top_n=1, aspect_min=0.40, fill_min=0.75,
                  collect_rejections=False):
    """Return top_n perforation candidates from a binary image as (cx, cy) tuples.

    Parameters
    ----------
    thresh : np.ndarray
        Binary image (uint8) to search for contours.
    roi_w : int
        Width of the ROI strip; used for the centroid-x position filter.
    top_n : int
        Maximum number of candidates to return.
    aspect_min : float
        Minimum aspect ratio (width/height) for a valid perforation contour.
        Use 0.40 for super8 (default); 0.25 for 8mm/super16.
    fill_min : float
        Minimum fill ratio (contour area / bounding rect area).
        Use 0.75 for super8 (default); 0.65 for 8mm/super16.
    collect_rejections : bool
        When True, also return a list of (x, y, bw, bh, reason_str) for each
        disqualified contour. This is used to generate annotated debug images.

    Return value matrix
    -------------------
    collect_rejections=False, top_n=1  → (cx, cy) or None
    collect_rejections=False, top_n>1  → list of (cx, cy), length ≤ top_n
    collect_rejections=True,  top_n=1  → ((cx, cy) or None, rejections_list)
    collect_rejections=True,  top_n>1  → (candidates_list, rejections_list)
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    rejections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)

        if area < 5000:
            if collect_rejections:
                rejections.append((x, y, bw, bh, "area"))
            continue

        # Compute centroid BEFORE the x-position filter so we use the true
        # centre of mass rather than the bounding rect left edge.
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + bw / 2.0
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else y + bh / 2.0

        if cx > roi_w * 0.70:
            if collect_rejections:
                rejections.append((x, y, bw, bh, "centroid-x"))
            continue

        aspect = bw / float(bh + 1e-6)
        if not (aspect_min <= aspect <= 1.20):
            if collect_rejections:
                rejections.append((x, y, bw, bh, "aspect"))
            continue

        fill_ratio = area / float((bw * bh) + 1e-6)
        if fill_ratio < fill_min:
            if collect_rejections:
                rejections.append((x, y, bw, bh, "fill"))
            continue

        score = area + (fill_ratio * 1000.0)
        candidates.append((score, (float(cx), float(cy))))

    candidates.sort(key=lambda c: c[0], reverse=True)
    results = [pt for _, pt in candidates[:top_n]]

    if collect_rejections:
        if top_n == 1:
            return (results[0] if results else None, rejections)
        return (results, rejections)
    else:
        if top_n == 1:
            return results[0] if results else None
        return results


def _save_debug_frame(roi_bgr, rejections, frame_name, debug_dir):
    """Save an annotated debug JPEG of the ROI crop for a failed detection.

    Draws rejected contour bounding rects in red labeled with the rejection
    reason, and overlays the source frame filename. Saved as quality-80 JPEG
    to keep file sizes small (~100 KB or less at typical scanner resolutions).

    Errors are caught silently so a non-writable debug_dir never crashes the
    main stabilisation run.

    Parameters
    ----------
    roi_bgr : np.ndarray
        BGR ROI crop (left strip of the source frame).
    rejections : list of (x, y, bw, bh, reason_str)
        Rejected contour info from _best_contour(collect_rejections=True).
    frame_name : str
        Source frame basename, used for the filename overlay and output name.
    debug_dir : str
        Directory where the debug JPEG is written.
    """
    try:
        img = roi_bgr.copy()
        h = img.shape[0]

        # Scale font to ROI height so labels are legible at any resolution.
        font_scale_reason = max(0.3, h / 3000.0)
        font_scale_name   = max(0.4, h / 2500.0)
        thickness = 1

        for rx, ry, rbw, rbh, reason in rejections:
            cv2.rectangle(img, (rx, ry), (rx + rbw, ry + rbh), (0, 0, 255), 2)
            label_y = max(ry - 4, int(font_scale_reason * 20))
            cv2.putText(img, reason, (rx, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_reason,
                        (0, 0, 255), thickness, cv2.LINE_AA)

        # Filename overlay in yellow at the top-left
        overlay_y = max(16, int(h * 0.015))
        cv2.putText(img, frame_name, (4, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_name,
                    (0, 255, 255), thickness, cv2.LINE_AA)

        basename  = os.path.splitext(frame_name)[0] + "_debug.jpg"
        out_path  = os.path.join(debug_dir, basename)
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    except Exception:
        # Never let a debug-image write error abort the stabilisation run.
        pass


def _annotate_roi_preview(roi_bgr, anchor, rejections, frame_name=''):
    """Return an annotated copy of an ROI-crop image for the UI preview panel.

    When anchor is not None (detection succeeded), draws a green crosshair and
    a "DETECTADO" label at the anchor position.  When anchor is None, draws
    rejected contour bounding rects in red with reason labels and a "NO
    DETECTADO" overlay — mirroring the _save_debug_frame style.

    Does NOT write to disk; the caller is responsible for saving.

    Parameters
    ----------
    roi_bgr : np.ndarray
        BGR ROI crop (left strip of the source frame).
    anchor : tuple(float, float) or None
        Detected anchor (cx, cy) in full-frame coordinates, or None.
    rejections : list of (x, y, bw, bh, reason_str)
        Rejected contour info from _best_contour(collect_rejections=True).
    frame_name : str
        Source frame basename used for the filename overlay label.

    Returns
    -------
    np.ndarray  annotated BGR image with same shape as roi_bgr.
    """
    img = roi_bgr.copy()
    h, w = img.shape[:2]

    font_scale = max(0.3, h / 3000.0)
    thickness  = 1

    # Frame-name overlay (top-left, yellow)
    if frame_name:
        overlay_y = max(16, int(h * 0.015))
        cv2.putText(img, frame_name, (4, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, h / 2500.0),
                    (0, 255, 255), thickness, cv2.LINE_AA)

    if anchor is not None:
        cx, cy = int(round(anchor[0])), int(round(anchor[1]))
        color_found = (0, 220, 0)  # green

        # Crosshair
        cv2.line(img, (0, cy),  (w, cy),  color_found, 1, cv2.LINE_AA)
        cv2.line(img, (cx, 0), (cx, h),  color_found, 1, cv2.LINE_AA)

        # Small filled circle at intersection
        cv2.circle(img, (cx, cy), max(4, int(h * 0.005)), color_found, -1, cv2.LINE_AA)

        # Label
        label_y = max(cy - 8, int(font_scale * 20))
        cv2.putText(img, "DETECTADO", (max(0, cx + 6), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color_found, thickness, cv2.LINE_AA)
    else:
        # Draw rejection boxes in red
        for rx, ry, rbw, rbh, reason in rejections:
            cv2.rectangle(img, (rx, ry), (rx + rbw, ry + rbh), (0, 0, 255), 2)
            label_y = max(ry - 4, int(font_scale * 20))
            cv2.putText(img, reason, (rx, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 255), thickness, cv2.LINE_AA)

        # "NO DETECTADO" banner
        banner_y = min(h - 8, int(h * 0.95))
        cv2.putText(img, "NO DETECTADO", (4, banner_y),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, h / 2000.0),
                    (0, 0, 255), 2, cv2.LINE_AA)

    return img


def detect_perforation(frame, roi_ratio=0.22, threshold=210, film_format='super8',
                       debug_dir=None, frame_name=''):
    """Detect the perforation anchor point in a frame.

    film_format values:
      'super8'  — 1 perf, left ROI (default)
      '8mm'     — 2 perfs per frame, left ROI; TOPMOST qualifying perf used as anchor
      'super16' — 2 perfs per frame, left ROI; TOPMOST qualifying perf used as anchor
    All formats scan the left side — perforations are always on the left.

    For '8mm' and 'super16', up to 2 candidates are found but the one with the
    smallest cy (closest to the top of the frame) is always returned.  This gives
    a consistent anchor regardless of how many perforations are visible in a given
    frame, eliminating the midpoint-vs-centroid inconsistency that caused large
    vertical anchor jumps when frame-to-frame detection count varied.

    When debug_dir and frame_name are provided, a failed detection (return None)
    causes an annotated ROI-crop JPEG to be written to debug_dir showing all
    rejected contours and their disqualifying reasons.
    """
    h, w = frame.shape[:2]
    roi_w = max(50, int(w * roi_ratio))
    kernel = np.ones((3, 3), np.uint8)

    def thresh_band(band_bgr):
        """Otsu-threshold a band; fall back to adaptive if Otsu saturates it.

        Triggers adaptive fallback when Otsu produces an all-black result
        (countNonZero == 0) OR an all-white / near-saturated result
        (countNonZero > 95 % of pixels) — both indicate Otsu failed to find
        a meaningful threshold.
        """
        gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, t = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel)
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)
        nz = cv2.countNonZero(t)
        if 0 < nz < int(0.95 * t.size):
            return t
        # Fallback: adaptive threshold for all-black OR near-all-white Otsu output
        bh, bw = band_bgr.shape[:2]
        block = max(51, (min(bh, roi_w) // 20) | 1)
        a = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, -5)
        a = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
        a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
        return a

    roi = frame[:, :roi_w]

    if film_format in ('8mm', 'super16'):
        # Single-pass full-ROI scan — no h//2 split.
        #
        # The previous approach bisected the frame at h//2 to isolate the two
        # perforations.  When a perforation fell near that line its contour was
        # split across both bands, causing both halves to fail the area/fill
        # filters and returning None (~10 % of frames for Diego's batches).
        #
        # Now we scan the full ROI at once, collect all qualifying contours,
        # pick the top-2 by score, and assign top/bottom roles by y-position.
        binary = thresh_band(roi)

        if debug_dir:
            candidates, rejections = _best_contour(
                binary, roi_w, top_n=2,
                aspect_min=0.25, fill_min=0.65,
                collect_rejections=True,
            )
        else:
            candidates = _best_contour(
                binary, roi_w, top_n=2,
                aspect_min=0.25, fill_min=0.65,
            )
            rejections = []

        if len(candidates) >= 1:
            # Always use the topmost qualifying perforation (smallest cy).
            # Scanning top_n=2 maximises detection recall, but we anchor to
            # the topmost candidate so the reference is consistent regardless
            # of whether 1 or 2 perfs are visible in a given frame.  Averaging
            # a midpoint caused large vertical anchor jumps when detection
            # count varied frame-to-frame.
            pt_top = min(candidates, key=lambda p: p[1])
            return (float(pt_top[0]), float(pt_top[1]))

        # No qualifying contour found.
        if debug_dir and frame_name:
            _save_debug_frame(roi, rejections, frame_name, debug_dir)
        return None

    # ── Super 8: single perforation ────────────────────────────────────────────
    # thresh_band() runs Otsu and falls back to adaptive when Otsu produces an
    # all-black OR near-saturated (>95 % white) result, covering overexposed frames.
    binary = thresh_band(roi)

    if debug_dir:
        pt, rejections = _best_contour(binary, roi_w, top_n=1, collect_rejections=True)
    else:
        pt = _best_contour(binary, roi_w, top_n=1)
        rejections = []

    if pt is not None:
        return (float(pt[0]), float(pt[1]))

    if debug_dir and frame_name:
        _save_debug_frame(roi, rejections, frame_name, debug_dir)
    return None


def stabilize_folder(input_dir, output_dir, progress_cb=None, log_cb=None,
                     roi_ratio=0.22, threshold=210, smooth_radius=9,
                     jpeg_quality=0, film_format='super8', debug_dir=None,
                     manual_anchor=None):
    files = list_images(input_dir)
    if not files:
        raise RuntimeError("No encontré imágenes dentro de la carpeta.")

    os.makedirs(output_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

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
            pt = detect_perforation(
                frame,
                roi_ratio=roi_ratio,
                threshold=threshold,
                film_format=film_format,
                debug_dir=debug_dir,
                frame_name=os.path.basename(f),
            )
            points.append(pt)
            if pt is None:
                failures += 1
                log(f"Sin detección: {os.path.basename(f)}")
        if progress_cb:
            progress_cb(i / (total * 2))

    valid = [p for p in points if p is not None]
    if not valid:
        raise RuntimeError("No logré detectar la perforación en ningún frame.")

    # Compute robust target: use manual anchor when provided, otherwise median of detections.
    if manual_anchor is not None:
        target_x = float(manual_anchor[0])
        target_y = float(manual_anchor[1])
        log(f"Usando referencia manual: x={target_x:.2f}, y={target_y:.2f}")
    else:
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
