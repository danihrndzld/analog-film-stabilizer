import glob
import os

import cv2
import numpy as np

from registration_stabilizer import (
    affine_from_components,
    components_from_matrix,
    crop_roi,
    detect_transform_breaks,
    estimate_registration_transform,
    identity_matrix,
    local_matrix_to_frame,
    preprocess_registration_image,
    resize_for_registration,
    scale_alignment_matrix,
    select_registration_roi,
    smooth_transform_components,
    texture_score,
)

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

_MIN_REGISTRATION_TEXTURE = 0.01


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

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


class _MotionPredictor:
    """Frame-to-frame position predictor with exponential-smoothed delta.

    Used by the tracking pipeline to pick the correct perforation among
    several top-K candidates: when two candidates score similarly, the one
    closest to ``predict()`` wins. The predictor intentionally does NOT
    update when a frame is flagged as ambiguous, so a bad match cannot
    contaminate future predictions (avoiding the cumulative-drift failure
    mode where one perf swap causes every subsequent frame to follow the
    wrong perforation).
    """

    def __init__(self, initial_pos, alpha=0.6):
        self.last_pos = (float(initial_pos[0]), float(initial_pos[1]))
        self.delta = (0.0, 0.0)
        self.alpha = float(alpha)

    def predict(self):
        return (
            self.last_pos[0] + self.delta[0],
            self.last_pos[1] + self.delta[1],
        )

    def update(self, new_pos):
        new_delta_x = new_pos[0] - self.last_pos[0]
        new_delta_y = new_pos[1] - self.last_pos[1]
        self.delta = (
            self.alpha * new_delta_x + (1.0 - self.alpha) * self.delta[0],
            self.alpha * new_delta_y + (1.0 - self.alpha) * self.delta[1],
        )
        self.last_pos = (float(new_pos[0]), float(new_pos[1]))


def _rank_candidates(
    candidates,
    predicted_pos,
    perf_spacing,
    ambiguity_score_ratio=0.85,
    ambiguity_spacing_tol=0.20,
):
    """Rank NCC candidates by combined score and flag perf-to-perf ambiguity.

    Combined score per candidate:

        score = ncc × exp(-distance_to_prediction / perf_spacing) × dominance

    where ``dominance`` divides this candidate's NCC by the maximum NCC
    among other candidates within ``perf_spacing`` — so a peak that stands
    alone ranks higher than one competing against a near-identical twin.

    The ambiguity flag fires when:
      - top-2 candidates have near-equal combined scores
        (score_2 >= ambiguity_score_ratio × score_1), AND
      - their positions are separated by ~perf_spacing
        (within ``ambiguity_spacing_tol``)

    This is the exact signature of "NCC locked onto a neighbouring
    perforation" — the caller should skip updating the motion predictor
    and reuse the predicted position for the frame.

    Returns
    -------
    (ranked, ambiguous) : (list, bool)
        ``ranked`` is the candidate list sorted by combined score
        descending. Each entry is ``(cx, cy, ncc, combined_score)``.
        ``ambiguous`` is True when perf-to-perf ambiguity is detected.
    """
    if not candidates:
        return [], False
    if perf_spacing is None or perf_spacing <= 0:
        # No spacing known: fall back to NCC-only ranking, no ambiguity
        ranked = sorted(
            ((cx, cy, ncc, ncc) for cx, cy, ncc in candidates),
            key=lambda t: t[3],
            reverse=True,
        )
        return list(ranked), False

    px, py = predicted_pos
    perf_spacing = float(perf_spacing)

    scored = []
    for cx, cy, ncc in candidates:
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        proximity = float(np.exp(-dist / perf_spacing))

        # Dominance: compare against other candidates within perf_spacing
        competing_ncc = 0.0
        for ox, oy, oncc in candidates:
            if ox == cx and oy == cy:
                continue
            odist = ((ox - cx) ** 2 + (oy - cy) ** 2) ** 0.5
            if odist <= perf_spacing and oncc > competing_ncc:
                competing_ncc = oncc
        dominance = ncc / competing_ncc if competing_ncc > 1e-6 else 1.0
        dominance = min(dominance, 1.0)

        combined = ncc * proximity * dominance
        scored.append((cx, cy, ncc, combined))

    scored.sort(key=lambda t: t[3], reverse=True)

    ambiguous = False
    if len(scored) >= 2:
        top = scored[0]
        runner = scored[1]
        if top[3] > 1e-6:
            score_close = runner[3] >= ambiguity_score_ratio * top[3]
        else:
            score_close = False
        sep = ((top[0] - runner[0]) ** 2 + (top[1] - runner[1]) ** 2) ** 0.5
        spacing_close = abs(sep - perf_spacing) <= ambiguity_spacing_tol * perf_spacing
        if score_close and spacing_close:
            ambiguous = True

    return scored, ambiguous


def _detect_perf_spacing(frame, anchor, search_radius=None):
    """Estimate vertical distance between adjacent perforations.

    Scans a vertical strip centred on the anchor for Otsu-bright contours
    whose size is similar to the anchor's perforation, and returns the
    median vertical centroid-to-centroid spacing. Falls back to None when
    fewer than two perforations are detected.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx, cy = anchor
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None

    bbox = _detect_perf_bbox(frame, anchor)
    if bbox is None:
        return None
    perf_w, perf_h = bbox
    if search_radius is None:
        search_radius = max(perf_h * 4, h // 3)

    # Narrow strip: just the perforation column. A wider strip catches bright
    # image content (sky, highlights) and pollutes the spacing estimate.
    half_w = max(int(perf_w * 0.6), 30)
    x0 = max(0, int(cx - half_w))
    x1 = min(w, int(cx + half_w))
    y0 = max(0, int(cy - search_radius))
    y1 = min(h, int(cy + search_radius))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    centers_y = []
    target_area = perf_w * perf_h
    target_ar = perf_w / max(perf_h, 1)
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 0.5 * target_area or area > 2.0 * target_area:
            continue
        ar = bw / max(bh, 1)
        if ar < target_ar * 0.6 or ar > target_ar * 1.6:
            continue
        centers_y.append(by + bh / 2.0)

    if len(centers_y) < 2:
        return None

    centers_y.sort()
    deltas = np.diff(centers_y)
    return float(np.median(deltas))


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

    half_h = int(min(1000, max(150, h * 0.3))) if half_h is None else int(half_h)

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


def _apply_affine_to_point(matrix, point):
    matrix = np.asarray(matrix, dtype=np.float32)
    x, y = point
    return (
        float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
        float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]),
    )


def _anchor_position_from_alignment(matrix, target):
    inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32))
    return _apply_affine_to_point(inverse, target)


def _registration_anchor_is_plausible(
    frame,
    matrix,
    target,
    reference_bbox,
    perf_spacing,
):
    h, w = frame.shape[:2]
    anchor_pos = _anchor_position_from_alignment(matrix, target)
    ax, ay = anchor_pos
    if not (np.isfinite(ax) and np.isfinite(ay)):
        return False, anchor_pos
    if ax < 0 or ay < 0 or ax >= w or ay >= h:
        return False, anchor_pos

    search_radius = int(max(80, min(perf_spacing * 0.35, min(h, w) * 0.25)))
    bbox = _detect_perf_bbox(frame, anchor_pos, search_radius=search_radius)
    if bbox is None:
        return False, anchor_pos
    if reference_bbox is None:
        return True, anchor_pos

    ref_w, ref_h = reference_bbox
    cur_w, cur_h = bbox
    width_ratio = cur_w / max(ref_w, 1)
    height_ratio = cur_h / max(ref_h, 1)
    return 0.45 <= width_ratio <= 2.2 and 0.45 <= height_ratio <= 2.2, anchor_pos


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
    """Stabilize a folder of frames using a registration strip around an anchor.

    Parameters
    ----------
    input_dir : str
        Path to folder containing input frames.
    output_dir : str
        Path to folder where stabilized frames will be written.
    anchor : tuple(float, float)
        User-selected (x, y) reference point in frame coordinates. The point seeds
        the film/perforation strip used for registration and remains the fallback
        tracker target.
    progress_cb : callable or None
        Called with a float 0.0-1.0 indicating progress.
    log_cb : callable or None
        Called with log message strings.
    smooth_radius : int
        Moving average window half-size for transform trajectory smoothing.
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

    failures = 0
    total = len(files)

    def log(msg):
        if log_cb:
            log_cb(msg)

    log(f"Encontré {total} imágenes.")

    # ── Phase 1: Build reference registration strip and fallback tracker ──
    log("Fase 1: preparando registro por franja de película...")

    first_frame = None
    for f in files:
        first_frame = cv2.imread(f)
        if first_frame is not None:
            break

    if first_frame is None:
        raise RuntimeError("No pude abrir ningún frame de la carpeta.")

    perf_bbox = _detect_perf_bbox(first_frame, anchor)
    registration_roi = select_registration_roi(first_frame.shape, anchor, perf_bbox)
    reference_strip = crop_roi(first_frame, registration_roi)
    reference_strip_small, registration_scale = resize_for_registration(
        reference_strip, max_dim=900
    )
    reference_processed = preprocess_registration_image(reference_strip_small)
    if (
        reference_processed.size == 0
        or texture_score(reference_processed) < _MIN_REGISTRATION_TEXTURE
    ):
        raise RuntimeError(
            "No pude construir una franja de registro con suficiente textura."
        )

    rx0, ry0, rx1, ry1 = registration_roi
    log(
        "Franja de registro: "
        f"x={rx0}:{rx1}, y={ry0}:{ry1} ({rx1 - rx0}×{ry1 - ry0} px, "
        f"escala {registration_scale:.3f})."
    )

    template, _ = _build_perforation_template(first_frame, anchor)
    if template is not None and float(np.std(template)) < 1e-6:
        template = None
    if template is None:
        log("No pude extraer plantilla NCC; el fallback por punto queda desactivado.")
    else:
        log(
            f"Plantilla fallback NCC ({template.shape[1]}×{template.shape[0]} px) "
            f"en ({anchor[0]:.0f}, {anchor[1]:.0f})"
        )

    # The user's anchor remains the fallback stabilization target.
    target_x = float(anchor[0])
    target_y = float(anchor[1])

    # Detect inter-perforation spacing from first frame. The registration path
    # does not require it, but the fallback tracker and discontinuity gate use it.
    perf_spacing = _detect_perf_spacing(first_frame, anchor)
    if perf_spacing is None:
        # Fallback: use frame-height heuristic (Regular 8mm has ~1 perf per frame)
        perf_spacing = float(first_frame.shape[0]) * 0.5
        log(
            f"No detecté espaciado entre perfs; usando heurística {perf_spacing:.0f}px."
        )
    else:
        log(f"Espaciado entre perforaciones detectado: {perf_spacing:.0f}px.")

    # Search radius for fallback NCC: generous enough to find the perf after
    # real jitter, but below the spacing when possible.
    search_radius = int(max(120, min(perf_spacing * 0.45, max(80, perf_spacing - 40))))
    predictor = _MotionPredictor(initial_pos=anchor, alpha=0.6)
    ambiguous_count = 0
    motion_rejected_count = 0

    tx_values = []
    ty_values = []
    theta_values = []
    valid_mask = []
    registration_confidences = []
    registration_success_count = 0
    ecc_count = 0
    phase_fallback_count = 0
    anchor_fallback_count = 0
    reused_transform_count = 0
    previous_matrix = identity_matrix()

    log("Fase 2: estimando trayectoria por registro de imagen...")

    for i, f in enumerate(files, 1):
        frame = cv2.imread(f)
        matrix = None
        method = "failed"
        confidence = 0.0
        ranked = []
        predicted = None

        if frame is None:
            failures += 1
            valid_mask.append(False)
            tx, ty, theta = components_from_matrix(previous_matrix)
            tx_values.append(tx)
            ty_values.append(ty)
            theta_values.append(theta)
            log(f"No pude abrir: {os.path.basename(f)}")
        else:
            current_strip = crop_roi(frame, registration_roi)
            current_strip_small, _ = resize_for_registration(current_strip, max_dim=900)
            current_processed = preprocess_registration_image(current_strip_small)
            max_translation = max(
                80.0,
                min(perf_spacing * 0.45, max(frame.shape[:2]) * 0.20),
            )
            result = estimate_registration_transform(
                reference_processed,
                current_processed,
                max_translation=max_translation * registration_scale,
            )

            if result.ok:
                local_matrix = scale_alignment_matrix(result.matrix, registration_scale)
                matrix = local_matrix_to_frame(local_matrix, registration_roi)
                plausible, registered_anchor = _registration_anchor_is_plausible(
                    frame,
                    matrix,
                    (target_x, target_y),
                    perf_bbox,
                    perf_spacing,
                )
                if plausible:
                    method = result.method
                    confidence = result.confidence
                    previous_matrix = matrix.copy()
                    predictor.update(registered_anchor)
                    registration_success_count += 1
                    registration_confidences.append(confidence)
                    if method == "ecc":
                        ecc_count += 1
                    elif method == "phase":
                        phase_fallback_count += 1
                else:
                    matrix = None
            if matrix is None and template is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                predicted = predictor.predict()
                candidates = _template_match_candidates(
                    gray,
                    template,
                    k=5,
                    search_center=predicted,
                    search_radius=search_radius,
                    min_confidence=0.3,
                )
                ranked, ambiguous = _rank_candidates(
                    candidates, predicted, perf_spacing
                )
                if ambiguous:
                    ambiguous_count += 1
                    motion_rejected_count += 1
                    log(
                        f"Registro falló y NCC ambiguo en {os.path.basename(f)}; "
                        "reuso trayectoria previa."
                    )
                elif ranked:
                    cx, cy, _ncc, _combined = ranked[0]
                    predictor.update((cx, cy))
                    matrix = np.float32([[1, 0, target_x - cx], [0, 1, target_y - cy]])
                    method = "anchor"
                    confidence = float(_ncc)
                    previous_matrix = matrix.copy()
                    anchor_fallback_count += 1

            if matrix is None:
                matrix = previous_matrix.copy()
                method = "reuse"
                failures += 1
                reused_transform_count += 1
                log(
                    f"Registro sin confianza en {os.path.basename(f)}; "
                    "reuso trayectoria previa."
                )

            tx, ty, theta = components_from_matrix(matrix)
            tx_values.append(tx)
            ty_values.append(ty)
            theta_values.append(theta)
            valid_mask.append(True)

            if debug_dir and method == "reuse":
                try:
                    basename = os.path.splitext(os.path.basename(f))[0] + "_debug.jpg"
                    debug_path = os.path.join(debug_dir, basename)
                    dbg = frame.copy()
                    cv2.rectangle(
                        dbg,
                        (registration_roi[0], registration_roi[1]),
                        (registration_roi[2] - 1, registration_roi[3] - 1),
                        (255, 0, 255),
                        3,
                    )
                    for rank_idx, c in enumerate(ranked[:5]):
                        cx_c, cy_c, ncc_c, _sc = c
                        color = (0, 255, 0) if rank_idx == 0 else (0, 165, 255)
                        cv2.circle(dbg, (int(cx_c), int(cy_c)), 20, color, 3)
                        cv2.putText(
                            dbg,
                            f"#{rank_idx + 1} {ncc_c:.2f}",
                            (int(cx_c) + 25, int(cy_c)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                    if predicted is not None:
                        px_p, py_p = predicted
                        cv2.drawMarker(
                            dbg,
                            (int(px_p), int(py_p)),
                            (255, 0, 255),
                            cv2.MARKER_CROSS,
                            30,
                            2,
                        )
                    cv2.putText(
                        dbg,
                        "registration fallback/reuse",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 255),
                        2,
                    )
                    cv2.imwrite(debug_path, dbg, [cv2.IMWRITE_JPEG_QUALITY, 80])
                except Exception as exc:
                    log(f"Debug image failed: {exc}")

        if progress_cb:
            progress_cb(i / (total * 2))

    valid_count = registration_success_count + anchor_fallback_count
    log(
        "Registro: "
        f"{registration_success_count} por imagen, "
        f"{anchor_fallback_count} por fallback NCC, "
        f"{reused_transform_count} reusados de {total}."
    )

    if valid_count == 0:
        raise RuntimeError("No logré estimar movimiento confiable en ningún frame.")

    log(f"Punto fijo objetivo: x={target_x:.2f}, y={target_y:.2f}")

    tx_arr = np.array(tx_values, dtype=np.float32)
    ty_arr = np.array(ty_values, dtype=np.float32)
    theta_arr = np.array(theta_values, dtype=np.float32)
    valid_arr = np.array(valid_mask, dtype=bool)
    breaks = detect_transform_breaks(
        tx_arr,
        ty_arr,
        theta_arr,
        valid_mask=valid_arr,
        translation_threshold=max(80.0, perf_spacing * 0.75),
        rotation_threshold=np.deg2rad(2.0),
    )
    smooth_radius = max(0, int(smooth_radius or 0))
    tx_s, ty_s, theta_s = smooth_transform_components(
        tx_arr,
        ty_arr,
        theta_arr,
        radius=smooth_radius,
        valid_mask=valid_arr,
        break_indices=breaks,
    )

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
            M = affine_from_components(tx_s[i - 1], ty_s[i - 1], theta_s[i - 1])
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

    mean_confidence = (
        float(np.mean(registration_confidences)) if registration_confidences else 0.0
    )

    summary = {
        "total_frames": total,
        "failed_detections": failures,
        "ambiguous_frames": ambiguous_count,
        "motion_rejected_frames": motion_rejected_count,
        "registration_success_frames": registration_success_count,
        "ecc_frames": ecc_count,
        "phase_fallback_frames": phase_fallback_count,
        "anchor_fallback_frames": anchor_fallback_count,
        "reused_transform_frames": reused_transform_count,
        "mean_registration_confidence": round(mean_confidence, 4),
        "registration_roi": {
            "x0": int(rx0),
            "y0": int(ry0),
            "x1": int(rx1),
            "y1": int(ry1),
            "width": int(rx1 - rx0),
            "height": int(ry1 - ry0),
        },
        "trajectory_segments": len(breaks),
        "smooth_radius": smooth_radius,
        "perf_spacing_px": round(float(perf_spacing), 1),
        "search_radius_px": int(search_radius),
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
