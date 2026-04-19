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


def _detect_perf_bbox(frame, anchor, search_radius=None):
    """Detect the perforation bounding box around a user-selected anchor.

    Uses Otsu thresholding on a grayscale ROI around the anchor, then picks
    the contour whose centroid is closest to the anchor (filtered by minimum
    area and aspect ratio to reject noise).

    Returns
    -------
    (w, h) : tuple(int, int) or None
        Width and height in pixels of the detected perforation, or None if
        no suitable contour is found.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cx, cy = anchor
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None

    if search_radius is None:
        search_radius = min(h, w) // 4

    x0 = max(0, int(cx - search_radius))
    y0 = max(0, int(cy - search_radius))
    x1 = min(w, int(cx + search_radius))
    y1 = min(h, int(cy + search_radius))

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    anchor_rx = cx - x0
    anchor_ry = cy - y0
    min_area = max(100, (search_radius * search_radius) // 400)

    best = None
    best_dist = float("inf")
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area:
            continue
        ar = bw / max(bh, 1)
        if ar < 0.3 or ar > 3.0:
            continue
        ccx = bx + bw / 2.0
        ccy = by + bh / 2.0
        d = (ccx - anchor_rx) ** 2 + (ccy - anchor_ry) ** 2
        if d < best_dist:
            best_dist = d
            best = (bw, bh)

    return best


def _build_perforation_template(frame, anchor, patch_radius=None):
    """Extract a tall grayscale patch around the user-selected anchor point.

    When ``patch_radius`` is None (default), the template size is auto-
    scaled to the detected perforation: width = ~1.2 × perf_w, height =
    ~2.5 × perf_h. The vertical asymmetry captures the perforation plus
    substantial dark film above and below — this is the context that lets
    downstream ranking tell "upper" vs "lower" perforation apart, instead
    of seeing two near-identical bright rectangles.

    When ``patch_radius`` is an int, behaves like the legacy square crop
    with that radius.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame.
    anchor : tuple(float, float)
        User-selected (x, y) in full-frame coordinates.
    patch_radius : int or None
        If None, auto-scale. Otherwise use as square half-size.

    Returns
    -------
    (template, origin) : (np.ndarray, (int, int))
        Grayscale template patch and the (x0, y0) origin of the patch in the
        frame. Returns (None, None) if extraction fails.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cx, cy = anchor
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None, None

    if patch_radius is None:
        bbox = _detect_perf_bbox(frame, anchor)
        if bbox is not None:
            perf_w, perf_h = bbox
            half_w = max(10, int(perf_w * 0.6))
            half_h = max(10, int(perf_h * 1.25))
        else:
            r = max(60, h // 10)
            half_w = r
            half_h = r
    else:
        half_w = int(patch_radius)
        half_h = int(patch_radius)

    x0 = max(0, int(cx - half_w))
    y0 = max(0, int(cy - half_h))
    x1 = min(w, int(cx + half_w))
    y1 = min(h, int(cy + half_h))

    if x1 - x0 < 10 or y1 - y0 < 10:
        return None, None

    template = gray[y0:y1, x0:x1].copy()
    return template, (x0, y0)


def _subpixel_refine(result, mx, my):
    """Parabolic sub-pixel refinement around an integer correlation peak.

    Returns refined (sx, sy) floats; falls back to the integer location
    when the quadratic fit is degenerate or the peak is at the boundary.
    """
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

    return sx, sy


def _extract_top_k_peaks(corr_map, k=5, suppression_radius=None):
    """Return the top-K local maxima of a correlation map.

    Iteratively finds the global maximum via ``cv2.minMaxLoc``, records it,
    and suppresses a square window of half-size ``suppression_radius``
    around the peak before repeating. Each returned peak is sub-pixel
    refined on the unmodified correlation surface.

    Parameters
    ----------
    corr_map : np.ndarray
        2D correlation surface (as produced by ``cv2.matchTemplate``).
    k : int
        Number of peaks to extract. Fewer may be returned if the map is
        exhausted by suppression before reaching K.
    suppression_radius : int or None
        Half-size of the square suppression window. Defaults to
        ``min(corr_map.shape) // 8`` when None.

    Returns
    -------
    list[tuple(float, float, float)]
        Entries are ``(sx, sy, score)`` in correlation-map coordinates
        (integer top-left + sub-pixel offset). Sorted by score descending.
    """
    rh, rw = corr_map.shape
    if suppression_radius is None:
        suppression_radius = max(1, min(rh, rw) // 8)
    suppression_radius = int(suppression_radius)

    # Preserve the original surface for sub-pixel refinement
    working = corr_map.copy()
    peaks = []
    very_low = float(corr_map.min()) - 1.0

    for _ in range(max(1, int(k))):
        _, max_val, _, max_loc = cv2.minMaxLoc(working)
        if not np.isfinite(max_val):
            break
        mx, my = int(max_loc[0]), int(max_loc[1])
        sx, sy = _subpixel_refine(corr_map, mx, my)
        peaks.append((sx, sy, float(max_val)))

        # Suppress a neighbourhood around this peak so the next iteration
        # picks a genuinely different local maximum
        sx0 = max(0, mx - suppression_radius)
        sy0 = max(0, my - suppression_radius)
        sx1 = min(rw, mx + suppression_radius + 1)
        sy1 = min(rh, my + suppression_radius + 1)
        working[sy0:sy1, sx0:sx1] = very_low

        # Bail out early if the remaining surface has nothing left to offer
        if float(working.max()) <= very_low:
            break

    return peaks


def _template_match_candidates(
    gray,
    template,
    k=5,
    search_center=None,
    search_radius=None,
    suppression_radius=None,
    min_confidence=0.0,
):
    """Locate up to K perforation candidates via NCC top-K peaks.

    Same ROI-restricted matching contract as ``_template_match_perforation``
    but returns a ranked list of candidates instead of the single global
    best. ``suppression_radius`` defaults to roughly half a template size
    so distinct perforations are returned as separate peaks.

    Returns
    -------
    list[tuple(float, float, float)]
        ``(cx, cy, ncc)`` entries in full-frame coordinates, sorted by NCC
        score descending. Empty list when the ROI is too small or no peak
        meets ``min_confidence``.
    """
    th, tw = template.shape[:2]
    h, w = gray.shape[:2]
    if h < th or w < tw:
        return []

    roi_x0 = 0
    roi_y0 = 0
    roi = gray

    if search_center is not None and search_radius is not None:
        sx_c, sy_c = search_center
        if np.isfinite(sx_c) and np.isfinite(sy_c):
            rx = int(search_radius) + tw // 2
            ry = int(search_radius) + th // 2
            x0 = max(0, int(sx_c) - rx)
            y0 = max(0, int(sy_c) - ry)
            x1 = min(w, int(sx_c) + rx)
            y1 = min(h, int(sy_c) + ry)
            if x1 - x0 >= tw and y1 - y0 >= th:
                roi_x0, roi_y0 = x0, y0
                roi = gray[y0:y1, x0:x1]

    corr = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)

    if suppression_radius is None:
        # Half a template size -- enough to force distinct perforations
        suppression_radius = max(th, tw) // 2

    peaks = _extract_top_k_peaks(corr, k=k, suppression_radius=suppression_radius)

    candidates = []
    for sx, sy, score in peaks:
        if score < min_confidence:
            continue
        cx = roi_x0 + sx + tw / 2.0
        cy = roi_y0 + sy + th / 2.0
        candidates.append((cx, cy, score))
    return candidates


def _template_match_perforation(
    gray,
    template,
    min_confidence=0.4,
    search_center=None,
    search_radius=None,
):
    """Locate the anchor point in a frame using normalized cross-correlation.

    Uses cv2.matchTemplate with TM_CCOEFF_NORMED and parabolic sub-pixel
    refinement on the correlation peak.

    When ``search_center`` and ``search_radius`` are provided, matching is
    restricted to an ROI around the expected position. This prevents the
    global NCC peak from locking onto a neighbouring (near-identical)
    perforation.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale frame.
    template : np.ndarray
        Grayscale template patch.
    min_confidence : float
        Minimum NCC score to accept the match (0.0-1.0).
    search_center : tuple(float, float) or None
        Expected (x, y) position of the anchor in frame coordinates. When
        given, search is limited to a box of radius ``search_radius``.
    search_radius : int or None
        Half-size (px) of the search box around ``search_center``. If the
        resulting ROI cannot contain the template, falls back to a full
        search.

    Returns
    -------
    (cx, cy, confidence) or None
        Sub-pixel (cx, cy) in frame coordinates and NCC confidence score,
        or None if the match is below min_confidence.
    """

    th, tw = template.shape[:2]
    h, w = gray.shape[:2]
    if h < th or w < tw:
        return None

    roi_x0 = 0
    roi_y0 = 0
    roi = gray

    if search_center is not None and search_radius is not None:
        sx_c, sy_c = search_center
        if np.isfinite(sx_c) and np.isfinite(sy_c):
            # Expand ROI by template half-size so the template fits around
            # the expected centre on all sides.
            rx = int(search_radius) + tw // 2
            ry = int(search_radius) + th // 2
            x0 = max(0, int(sx_c) - rx)
            y0 = max(0, int(sy_c) - ry)
            x1 = min(w, int(sx_c) + rx)
            y1 = min(h, int(sy_c) + ry)
            if x1 - x0 >= tw and y1 - y0 >= th:
                roi_x0, roi_y0 = x0, y0
                roi = gray[y0:y1, x0:x1]

    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < min_confidence:
        return None

    mx, my = max_loc  # top-left of best match within ROI (integer)

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

    # Convert from ROI match top-left to template center in frame coords
    cx = roi_x0 + sx + tw / 2.0
    cy = roi_y0 + sy + th / 2.0

    return (cx, cy, max_val)


def _build_rotation_template(frame, anchor, half_w=None, half_h=None):
    """Extract a tall grayscale strip around the anchor for rotation ECC.

    A taller crop gives ECC more vertical lever arm, so sub-tenth-degree
    rotation is estimated more stably than from a square patch. When
    ``half_h`` is None, it is auto-scaled to ~30% of the frame height
    (capped at 1000 px to bound ECC memory) so high-resolution scans
    (e.g. 4056×3040) get meaningful rotational resolution instead of the
    legacy 300-px strip. ``half_w`` defaults to the detected perforation
    width, scaling horizontal context with the scan resolution.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cx, cy = anchor
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None

    if half_h is None:
        half_h = int(min(1000, max(150, h * 0.3)))
    else:
        half_h = int(half_h)

    if half_w is None:
        bbox = _detect_perf_bbox(frame, anchor)
        if bbox is not None:
            perf_w, _ = bbox
            half_w = max(60, int(perf_w * 0.6))
        else:
            half_w = 60
    else:
        half_w = int(half_w)

    x0 = max(0, int(cx - half_w))
    y0 = max(0, int(cy - half_h))
    x1 = min(w, int(cx + half_w))
    y1 = min(h, int(cy + half_h))

    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    return gray[y0:y1, x0:x1].copy()


def _estimate_rotation(gray, template, cx, cy):
    """Estimate small rotation angle of the anchor region using ECC.

    Uses cv2.findTransformECC with MOTION_EUCLIDEAN (translation + rotation).
    ``template`` should typically be a tall strip around the anchor (see
    ``_build_rotation_template``) so ECC has vertical lever arm.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale frame.
    template : np.ndarray
        Grayscale reference crop (commonly the rotation strip).
    cx, cy : float
        Template-matched center position of the anchor in frame coordinates.

    Returns
    -------
    float or None
        Rotation angle in degrees (positive = counter-clockwise), or None
        when ECC cannot be computed (out of bounds, convergence failure).
    """
    h, w = gray.shape[:2]

    th, tw = template.shape[:2]
    x0 = int(round(cx - tw / 2.0))
    y0 = int(round(cy - th / 2.0))
    x1 = x0 + tw
    y1 = y0 + th

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    patch = gray[y0:y1, x0:x1]

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
        # For EUCLIDEAN: [[cos θ, -sin θ, tx], [sin θ, cos θ, ty]]
        angle_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        return float(np.degrees(angle_rad))
    except cv2.error:
        return None


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

    rot_template = _build_rotation_template(first_frame, anchor)

    log(
        f"Plantilla construida ({template.shape[1]}×{template.shape[0]} px) "
        f"en ({anchor[0]:.0f}, {anchor[1]:.0f})"
    )

    # The user's anchor IS the stabilization target
    target_x = float(anchor[0])
    target_y = float(anchor[1])

    # ── Phase 2: Detect anchor position in all frames via template matching ──
    log("Fase 2: alineando frames...")

    # Tracking state: seed search at the user anchor; update as we find matches.
    # ROI search keeps NCC from locking onto an adjacent perforation. The
    # motion-prior threshold rejects implausible jumps (also typically a
    # neighbouring perf) and forces the position through the interpolation
    # fallback.
    search_radius = 220  # px; covers realistic frame-to-frame drift
    max_jump = 180  # px; above this, treat match as neighbouring perf
    last_pos = (float(anchor[0]), float(anchor[1]))

    for i, f in enumerate(files, 1):
        frame = cv2.imread(f)
        if frame is None:
            points.append(None)
            angles.append(None)
            failures += 1
            log(f"No pude abrir: {os.path.basename(f)}")
        else:
            pt = None
            angle = None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tm_result = _template_match_perforation(
                gray,
                template,
                min_confidence=0.4,
                search_center=last_pos,
                search_radius=search_radius,
            )
            if tm_result is not None:
                cx, cy, _conf = tm_result
                jump = ((cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2) ** 0.5
                if jump > max_jump:
                    log(
                        f"Salto implausible ({jump:.0f}px) en "
                        f"{os.path.basename(f)}; se interpolará."
                    )
                else:
                    pt = (cx, cy)
                    last_pos = pt
                    if rot_template is not None:
                        angle = _estimate_rotation(gray, rot_template, cx, cy)

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

    # Interpolate missing angles (ECC failures, out-of-bounds) then smooth.
    # Raw per-frame ECC on a small patch is noisy at the sub-tenth-degree
    # scale; without smoothing that noise warps every frame slightly.
    ang = np.array(
        [a if a is not None else np.nan for a in angles], dtype=np.float32
    )
    idx_a = np.arange(len(ang))
    good_a = np.isfinite(ang)
    if np.any(good_a):
        ang[~good_a] = np.interp(idx_a[~good_a], idx_a[good_a], ang[good_a])
    else:
        ang[:] = 0.0
    k = smooth_radius * 2 + 1
    kernel = np.ones(k, dtype=np.float32) / k
    ang_smoothed = np.convolve(
        np.pad(ang, (smooth_radius, smooth_radius), mode="edge"),
        kernel,
        mode="valid",
    )
    per_frame_angles = ang_smoothed.tolist()

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
            angle = (
                per_frame_angles[i - 1] if i - 1 < len(per_frame_angles) else 0.0
            )

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
