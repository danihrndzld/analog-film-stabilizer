import glob
import os

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
    return list(zip(xs_s.tolist(), ys_s.tolist(), strict=True))


_BORDER_MODES = {
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT_101,
}


# ── Template matching alignment ──────────────────────────────────────────────


def _build_perforation_template(frame, anchor, patch_radius=60):
    """Extract a grayscale patch around the user-selected anchor point.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame.
    anchor : tuple(float, float)
        User-selected (x, y) in full-frame coordinates.
    patch_radius : int
        Radius in pixels around the anchor to extract.

    Returns
    -------
    (template, origin) : (np.ndarray, (int, int))
        Grayscale template patch and the (x0, y0) origin of the patch in the frame.
        Returns (None, None) if extraction fails.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cx, cy = anchor
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None, None

    x0 = max(0, int(cx - patch_radius))
    y0 = max(0, int(cy - patch_radius))
    x1 = min(w, int(cx + patch_radius))
    y1 = min(h, int(cy + patch_radius))

    if x1 - x0 < 10 or y1 - y0 < 10:
        return None, None

    template = gray[y0:y1, x0:x1].copy()
    return template, (x0, y0)


def _template_match_perforation(gray, template, min_confidence=0.4):
    """Locate the anchor point in a frame using normalized cross-correlation.

    Uses cv2.matchTemplate with TM_CCOEFF_NORMED and parabolic sub-pixel
    refinement on the correlation peak.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale frame.
    template : np.ndarray
        Grayscale template patch.
    min_confidence : float
        Minimum NCC score to accept the match (0.0-1.0).

    Returns
    -------
    (cx, cy, confidence) or None
        Sub-pixel (cx, cy) in frame coordinates and NCC confidence score,
        or None if the match is below min_confidence.
    """

    th, tw = template.shape[:2]
    if gray.shape[0] < th or gray.shape[1] < tw:
        return None

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < min_confidence:
        return None

    mx, my = max_loc  # top-left of best match (integer)

    # Sub-pixel refinement via parabolic interpolation on the correlation surface
    rh, rw = result.shape
    sx, sy = float(mx), float(my)

    if 0 < mx < rw - 1:
        left = float(result[my, mx - 1])
        center = float(result[my, mx])
        right = float(result[my, mx + 1])
        denom = 2.0 * (2.0 * center - left - right)
        if abs(denom) > 1e-6:
            sx = mx + (left - right) / denom

    if 0 < my < rh - 1:
        top = float(result[my - 1, mx])
        center = float(result[my, mx])
        bottom = float(result[my + 1, mx])
        denom = 2.0 * (2.0 * center - top - bottom)
        if abs(denom) > 1e-6:
            sy = my + (top - bottom) / denom

    # Convert from match top-left to template center
    cx = sx + tw / 2.0
    cy = sy + th / 2.0

    return (cx, cy, max_val)


def _estimate_rotation(gray, template, cx, cy):
    """Estimate small rotation angle of the anchor region using ECC.

    Uses cv2.findTransformECC with MOTION_EUCLIDEAN (translation + rotation)
    on a tight crop around the template-matched position.  Returns the rotation
    angle in degrees, or 0.0 if ECC fails to converge.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale frame.
    template : np.ndarray
        Grayscale template patch.
    cx, cy : float
        Template-matched center position in frame coordinates.

    Returns
    -------
    float
        Rotation angle in degrees (positive = counter-clockwise).
    """
    h, w = gray.shape[:2]

    th, tw = template.shape[:2]
    # Extract the matched region from the current frame
    x0 = max(0, int(cx - tw / 2))
    y0 = max(0, int(cy - th / 2))
    x1 = min(w, x0 + tw)
    y1 = min(h, y0 + th)

    if x1 - x0 != tw or y1 - y0 != th:
        return 0.0

    patch = gray[y0:y1, x0:x1]

    # ECC with EUCLIDEAN model (translation + rotation)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)

    try:
        _, warp_matrix = cv2.findTransformECC(
            template.astype(np.float32),
            patch.astype(np.float32),
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
        )
        # Extract rotation angle from the 2×3 matrix
        # For EUCLIDEAN: [[cos θ, -sin θ, tx], [sin θ, cos θ, ty]]
        angle_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        return float(np.degrees(angle_rad))
    except cv2.error:
        return 0.0


def stabilize_folder(
    input_dir,
    output_dir,
    anchor,
    progress_cb=None,
    log_cb=None,
    smooth_radius=9,
    jpeg_quality=0,
    debug_dir=None,
    border_mode="replicate",
):
    """Stabilize a folder of frames using the user-selected anchor point.

    Parameters
    ----------
    input_dir : str
        Path to folder containing input frames.
    output_dir : str
        Path to folder where stabilized frames will be written.
    anchor : tuple(float, float)
        User-selected (x, y) reference point in frame coordinates.
        This is the point that will be locked to a fixed position.
    progress_cb : callable or None
        Called with a float 0.0-1.0 indicating progress.
    log_cb : callable or None
        Called with log message strings.
    smooth_radius : int
        Moving average window half-size for position smoothing.
    jpeg_quality : int
        JPEG quality 1-100, or 0 for PNG lossless output.
    debug_dir : str or None
        Directory for debug images on failed detections.
    border_mode : str
        Border fill mode: 'replicate', 'constant', or 'reflect'.
    """
    if anchor is None:
        raise ValueError("Se requiere un punto de referencia (anchor).")

    files = list_images(input_dir)
    if not files:
        raise RuntimeError("No encontré imágenes dentro de la carpeta.")

    os.makedirs(output_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    points = []
    angles = []  # per-frame rotation angles (degrees)
    failures = 0
    total = len(files)

    def log(msg):
        if log_cb:
            log_cb(msg)

    log(f"Encontré {total} imágenes.")

    # ── Phase 1: Build reference template from user-selected anchor ──
    log("Fase 1: construyendo plantilla de referencia...")

    first_frame = None
    for f in files:
        first_frame = cv2.imread(f)
        if first_frame is not None:
            break

    if first_frame is None:
        raise RuntimeError("No pude abrir ningún frame de la carpeta.")

    template, _ = _build_perforation_template(first_frame, anchor)
    if template is None:
        raise RuntimeError(
            f"No pude extraer plantilla en el punto ({anchor[0]:.0f}, {anchor[1]:.0f}). "
            "Selecciona un punto con más contraste."
        )

    log(
        f"Plantilla construida ({template.shape[1]}×{template.shape[0]} px) "
        f"en ({anchor[0]:.0f}, {anchor[1]:.0f})"
    )

    # The user's anchor IS the stabilization target
    target_x = float(anchor[0])
    target_y = float(anchor[1])

    # ── Phase 2: Detect anchor position in all frames via template matching ──
    log("Fase 2: alineando frames...")

    for i, f in enumerate(files, 1):
        frame = cv2.imread(f)
        if frame is None:
            points.append(None)
            angles.append(0.0)
            failures += 1
            log(f"No pude abrir: {os.path.basename(f)}")
        else:
            pt = None
            angle = 0.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tm_result = _template_match_perforation(gray, template, min_confidence=0.4)
            if tm_result is not None:
                cx, cy, conf = tm_result
                pt = (cx, cy)
                angle = _estimate_rotation(gray, template, cx, cy)

            points.append(pt)
            angles.append(angle)
            if pt is None:
                failures += 1
                log(f"Sin detección: {os.path.basename(f)}")
                # Save debug image if requested
                if debug_dir:
                    try:
                        basename = (
                            os.path.splitext(os.path.basename(f))[0] + "_debug.jpg"
                        )
                        debug_path = os.path.join(debug_dir, basename)
                        h_f, w_f = frame.shape[:2]
                        th, tw = template.shape[:2]
                        pr_y = th // 2
                        pr_x = tw // 2
                        x0 = max(0, int(anchor[0]) - pr_x)
                        y0 = max(0, int(anchor[1]) - pr_y)
                        x1 = min(w_f, int(anchor[0]) + pr_x)
                        y1 = min(h_f, int(anchor[1]) + pr_y)
                        patch = frame[y0:y1, x0:x1].copy()
                        cv2.imwrite(debug_path, patch, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    except Exception as exc:
                        log(f"Debug image failed: {exc}")
        if progress_cb:
            progress_cb(i / (total * 2))

    log(f"Detección: {total - failures} exitosos, {failures} fallidos de {total}")

    valid = [p for p in points if p is not None]
    if not valid:
        raise RuntimeError("No logré detectar la referencia en ningún frame.")

    log(f"Punto fijo objetivo: x={target_x:.2f}, y={target_y:.2f}")

    # Fill None positions (failed detections) by linear interpolation.
    xs = np.array([p[0] if p is not None else np.nan for p in points], dtype=np.float32)
    ys = np.array([p[1] if p is not None else np.nan for p in points], dtype=np.float32)
    idx = np.arange(len(xs))
    good_x = np.isfinite(xs)
    good_y = np.isfinite(ys)
    if np.any(good_x):
        xs[~good_x] = np.interp(idx[~good_x], idx[good_x], xs[good_x])
    if np.any(good_y):
        ys[~good_y] = np.interp(idx[~good_y], idx[good_y], ys[good_y])
    per_frame = list(zip(xs.tolist(), ys.tolist(), strict=True))

    log("Segunda pasada: estabilizando y guardando...")

    cv_border = _BORDER_MODES.get(border_mode, cv2.BORDER_REPLICATE)
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
            angle = angles[i - 1] if i - 1 < len(angles) else 0.0

            # Build rotation + translation matrix.
            # Rotate around the detected position to correct in-gate tilt,
            # then translate to align to the target.
            if abs(angle) > 0.001:
                perf_x, perf_y = pt[0], pt[1]
                R = cv2.getRotationMatrix2D((perf_x, perf_y), -angle, 1.0)
                R[0, 2] += dx
                R[1, 2] += dy
                M = R
            else:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
            stabilized = cv2.warpAffine(
                frame,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv_border,
            )

            basename = os.path.basename(f)
            if jpeg_quality == 0:
                basename = os.path.splitext(basename)[0] + ".png"
                out_path = os.path.join(output_dir, basename)
                cv2.imwrite(out_path, stabilized)
            else:
                out_path = os.path.join(output_dir, basename)
                cv2.imwrite(
                    out_path, stabilized, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
                )
        if progress_cb:
            progress_cb((total + i) / (total * 2))

    summary = {
        "total_frames": total,
        "failed_detections": failures,
        "target_x": round(target_x, 3),
        "target_y": round(target_y, 3),
        "output_width": out_w,
        "output_height": out_h,
        "output_format": "png (lossless)"
        if jpeg_quality == 0
        else f"jpeg q{jpeg_quality}",
        "border_mode": border_mode,
    }

    with open(
        os.path.join(output_dir, "stabilization_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("PERFORATION STABILIZATION REPORT\n")
        f.write("================================\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    log("Listo.")
    log(f"Frames: {summary['total_frames']}")
    log(f"Sin detección: {summary['failed_detections']}")
    log(f"Tamaño de salida: {out_w}×{out_h} px")
    return summary
