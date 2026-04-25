import glob
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from trajectory_smoothing import detect_splices, rigid_fit_2pt, smooth_trajectory

VALID_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")


def list_images(folder):
    files = set()
    for ext in VALID_EXTS:
        files.update(glob.glob(os.path.join(folder, ext)))
        files.update(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(files)


_BORDER_MODES = {
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT_101,
}


_FrameOutcome = namedtuple(
    "_FrameOutcome", "pt ambiguous motion_rejected ranked predicted"
)


@dataclass
class TrackPassResult:
    """Per-frame tracking outputs for a single anchor's pass.

    Phase A fields only. ``consensus_rejected`` is initialized to all-False
    here and populated by the consensus-gate stage (Unit 5). Phase B will
    extend this dataclass with ``pass1_ncc_at_pos`` when same-window-size
    NCC comparison ships (Unit 10); deferred until that consumer exists to
    avoid dead-field drift if R14 ships Phase A only.
    """

    positions: list[tuple[float, float] | None]
    nccs: np.ndarray  # float64, NaN where no candidate
    ambiguous_mask: np.ndarray  # bool
    motion_rejected: np.ndarray  # bool
    consensus_rejected: np.ndarray  # bool, populated in Unit 5
    failed_mask: np.ndarray  # bool
    outcomes: list[Any]  # list[_FrameOutcome | None]


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

    Returns
    -------
    (template, anchor_in_tpl) : (np.ndarray, (float, float))
        Grayscale template patch and the (ax, ay) sub-pixel position of the
        anchor within the template. Returns (None, None) if extraction fails.
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
    anchor_in_tpl = (float(cx) - x0, float(cy) - y0)
    return template, anchor_in_tpl


def _extract_top_k_peaks(corr_map, k=5, suppression_radius=None):
    """Return the top-K local maxima of a correlation map.

    Iteratively finds the global maximum via ``cv2.minMaxLoc``, records it,
    and suppresses a square window of half-size ``suppression_radius``
    around the peak before repeating. Each peak is sub-pixel refined using
    the current surface; if a ±1 neighbour was already poisoned by a prior
    suppression window, that axis falls back to the integer peak.

    Note: ``corr_map`` is consumed in place to avoid allocating a copy on
    every frame (this is on the hot path).

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

    peaks = []
    very_low = float(corr_map.min()) - 1.0
    guard = very_low + 0.5  # anything <= guard is a poisoned neighbour

    for _ in range(max(1, int(k))):
        _, max_val, _, max_loc = cv2.minMaxLoc(corr_map)
        if not np.isfinite(max_val) or max_val <= guard:
            break
        mx, my = int(max_loc[0]), int(max_loc[1])

        # Guarded sub-pixel refine: skip axis if neighbour was suppressed
        sx, sy = float(mx), float(my)
        if 0 < mx < rw - 1:
            left = float(corr_map[my, mx - 1])
            right = float(corr_map[my, mx + 1])
            if left > guard and right > guard:
                center = float(corr_map[my, mx])
                denom = 2.0 * (2.0 * center - left - right)
                if abs(denom) > 1e-6:
                    sx = mx + (left - right) / denom
        if 0 < my < rh - 1:
            top = float(corr_map[my - 1, mx])
            bottom = float(corr_map[my + 1, mx])
            if top > guard and bottom > guard:
                center = float(corr_map[my, mx])
                denom = 2.0 * (2.0 * center - top - bottom)
                if abs(denom) > 1e-6:
                    sy = my + (top - bottom) / denom

        peaks.append((sx, sy, float(max_val)))

        sx0 = max(0, mx - suppression_radius)
        sy0 = max(0, my - suppression_radius)
        sx1 = min(rw, mx + suppression_radius + 1)
        sy1 = min(rh, my + suppression_radius + 1)
        corr_map[sy0:sy1, sx0:sx1] = very_low

    return peaks


def _template_match_candidates(
    gray,
    template,
    k=5,
    search_center=None,
    search_radius=None,
    suppression_radius=None,
    min_confidence=0.0,
    anchor_in_tpl=None,
):
    """Locate up to K perforation candidates via NCC top-K peaks.

    ``anchor_in_tpl`` is the (ax, ay) position of the anchor within the
    template; when None, defaults to the template centre (legacy behaviour).
    Returned candidate coordinates are the anchor position in full-frame
    coordinates, not the template centre.

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

    if anchor_in_tpl is None:
        ax, ay = tw / 2.0, th / 2.0
    else:
        ax, ay = float(anchor_in_tpl[0]), float(anchor_in_tpl[1])

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
        cx = roi_x0 + sx + ax
        cy = roi_y0 + sy + ay
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

    Returns
    -------
    (ranked, ambiguous) : (list, bool)
        ``ranked`` is the candidate list sorted by combined score
        descending. Each entry is ``(cx, cy, ncc, combined_score)``.
    """
    if not candidates:
        return [], False
    if perf_spacing is None or perf_spacing <= 0:
        ranked = sorted(
            ((cx, cy, ncc, ncc) for cx, cy, ncc in candidates),
            key=lambda t: t[3],
            reverse=True,
        )
        return list(ranked), False

    px, py = predicted_pos
    perf_spacing = float(perf_spacing)

    scored = []
    for i, (cx, cy, ncc) in enumerate(candidates):
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        proximity = float(np.exp(-dist / perf_spacing))

        # Dominance: compare against other candidates (by index) within perf_spacing
        competing_ncc = 0.0
        for j, (ox, oy, oncc) in enumerate(candidates):
            if j == i:
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
    median vertical centroid-to-centroid spacing. Returns None when fewer
    than three perforations are detected or when the median spacing is
    less than 1.5× the perforation height (spurious close-together
    contours from e.g. sprocket shadow fragments).
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

    # Require at least 3 centres: two centres yield a single delta, which is
    # easy to fake with one spurious contour. Three centres give two deltas
    # and a median that actually reflects repetition.
    if len(centers_y) < 3:
        return None

    centers_y.sort()
    deltas = np.diff(centers_y)
    median_delta = float(np.median(deltas))

    # Reject implausibly tight spacings: real inter-perf spacing is several
    # times the perforation height. Anything closer is a contour-fragment
    # artefact, not genuine periodicity.
    if median_delta < 1.5 * perf_h:
        return None

    return median_delta


def _locate_anchor_in_frame(
    frame,
    template,
    anchor_in_tpl,
    predictor,
    search_radius,
    perf_spacing,
    min_confidence=0.3,
):
    """Locate the anchor in one frame. Returns a _FrameOutcome.

    Outcome semantics:
      - success: ``pt`` is the detected (cx, cy); predictor updated by caller
      - ambiguous: perf-to-perf ambiguity. ``pt`` is the predicted position
        (used for this frame's warp so the sequence does not jump), but the
        predictor is NOT updated — a swapped perf must not contaminate
        future predictions
      - motion_rejected: no candidates in the search ROI at all. ``pt`` is
        None; the frame will be interpolated from neighbours
      - no_candidates: candidates exist but all below min_confidence;
        treated as motion_rejected
    """
    predicted = predictor.predict()

    # Crop BGR ROI around the prediction first, then grayscale only that ROI.
    # Whole-frame grayscale on a 4056×3040 scan is pure waste — we only search
    # inside ±search_radius anyway.
    h, w = frame.shape[:2]
    th, tw = template.shape[:2]
    px, py = predicted
    if np.isfinite(px) and np.isfinite(py):
        rx = int(search_radius) + tw
        ry = int(search_radius) + th
        x0 = max(0, int(px) - rx)
        y0 = max(0, int(py) - ry)
        x1 = min(w, int(px) + rx)
        y1 = min(h, int(py) + ry)
        roi_bgr = frame[y0:y1, x0:x1]
        if roi_bgr.size == 0 or x1 - x0 < tw or y1 - y0 < th:
            return _FrameOutcome(None, False, True, [], predicted)
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        # Search within the ROI; candidate coords come back in ROI space,
        # then we translate to full-frame space.
        roi_center = (px - x0, py - y0)
        candidates = _template_match_candidates(
            gray_roi,
            template,
            k=5,
            search_center=roi_center,
            search_radius=search_radius,
            min_confidence=min_confidence,
            anchor_in_tpl=anchor_in_tpl,
        )
        candidates = [(cx + x0, cy + y0, ncc) for cx, cy, ncc in candidates]
    else:
        return _FrameOutcome(None, False, True, [], predicted)

    if not candidates:
        return _FrameOutcome(None, False, True, [], predicted)

    ranked, ambiguous = _rank_candidates(candidates, predicted, perf_spacing)
    if ambiguous:
        # Reuse prediction for the warp so the output frame isn't blank,
        # but don't update the predictor — we don't trust this position.
        return _FrameOutcome(predicted, True, False, ranked, predicted)

    if ranked:
        cx, cy, _ncc, _combined = ranked[0]
        # Motion-plausibility gate: a lone top candidate that sits more than
        # half a perforation-spacing away from where the predictor expects us
        # is almost certainly a neighbour-perf swap. Reject it so the
        # predictor doesn't lock onto the wrong perforation.
        if perf_spacing is not None:
            px, py = predicted
            d = float(np.hypot(cx - px, cy - py))
            if d > 0.5 * perf_spacing:
                return _FrameOutcome(None, False, True, ranked, predicted)
        return _FrameOutcome((cx, cy), False, False, ranked, predicted)

    return _FrameOutcome(None, False, True, ranked, predicted)


MIN_ANCHOR_SEPARATION_FRAC = 0.25


def _track_anchor_pass(
    files,
    template,
    anchor_in_tpl,
    predictor,
    search_radius,
    perf_spacing,
    log,
    label,
    calibration=None,
):
    """Run per-frame tracking for a single anchor across all frames.

    Returns a ``TrackPassResult`` dataclass with parallel arrays of length
    ``len(files)``:
      - positions: (x, y) tuples or None (None = ambiguous/rejected/failed)
      - nccs: top-candidate NCC (or NaN when no candidate)
      - ambiguous_mask: bool per frame
      - motion_rejected: bool per frame (predictor-distance gate)
      - consensus_rejected: bool per frame, all-False here (populated by
        Unit 5's R4 consensus check, which runs after both anchor passes)
      - failed_mask: bool per frame (frame unreadable or no candidates)
      - outcomes: the raw _FrameOutcome (for debug drawing)

    ``calibration`` is reserved for downstream consumers (Unit 5/8 will
    consume it for R4 gates and R13 health-check inputs); it is currently
    accepted but unused by the tracking loop itself.
    """
    n = len(files)
    positions: list[tuple[float, float] | None] = [None] * n
    nccs = np.full(n, np.nan, dtype=np.float64)
    amb = np.zeros(n, dtype=bool)
    rej = np.zeros(n, dtype=bool)
    fail = np.zeros(n, dtype=bool)
    outcomes: list[Any] = [None] * n

    for i, f in enumerate(files):
        frame = cv2.imread(f)
        if frame is None:
            fail[i] = True
            log(f"[{label}] No pude abrir: {os.path.basename(f)}")
            continue

        outcome = _locate_anchor_in_frame(
            frame,
            template,
            anchor_in_tpl,
            predictor,
            search_radius,
            perf_spacing,
            min_confidence=0.3,
        )
        outcomes[i] = outcome
        if outcome.ranked:
            nccs[i] = float(outcome.ranked[0][2])

        if outcome.ambiguous:
            amb[i] = True
        elif outcome.motion_rejected:
            rej[i] = True
        elif outcome.pt is not None:
            predictor.update(outcome.pt)
            positions[i] = outcome.pt
        else:
            fail[i] = True

    return TrackPassResult(
        positions=positions,
        nccs=nccs,
        ambiguous_mask=amb,
        motion_rejected=rej,
        consensus_rejected=np.zeros(n, dtype=bool),
        failed_mask=fail,
        outcomes=outcomes,
    )


def stabilize_folder(
    input_dir,
    output_dir,
    anchor1,
    anchor2,
    progress_cb=None,
    log_cb=None,
    jpeg_quality=0,
    debug_dir=None,
    border_mode="replicate",
):
    """Stabilize a folder of frames using two user-selected anchor points.

    Two anchors give a rotation lever arm plus detection redundancy. Each
    anchor is tracked independently; frames where both anchors succeed get a
    closed-form rigid transform (translation + rotation). The full transform
    series is then globally smoothed with splice-segmented Savitzky-Golay +
    MAD outlier refit before warp.

    Parameters
    ----------
    input_dir, output_dir : str
    anchor1, anchor2 : tuple(float, float)
        User-selected reference points. Both required.
    progress_cb : callable or None
    log_cb : callable or None
    jpeg_quality : int  -- 0 for PNG lossless, 1-100 for JPEG
    debug_dir : str or None
    border_mode : str -- 'replicate', 'constant', or 'reflect'
    """
    if anchor1 is None or anchor2 is None:
        raise ValueError("Se requieren dos puntos de referencia (anchor1 y anchor2).")

    files = list_images(input_dir)
    if not files:
        raise RuntimeError("No encontré imágenes dentro de la carpeta.")

    os.makedirs(output_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    total = len(files)

    def log(msg):
        if log_cb:
            log_cb(msg)

    log(f"Encontré {total} imágenes.")

    # ── Phase 1: Build reference templates and validate anchor separation ──
    log("Fase 1: construyendo plantillas de referencia...")

    first_frame = None
    for f in files:
        first_frame = cv2.imread(f)
        if first_frame is not None:
            break
    if first_frame is None:
        raise RuntimeError("No pude abrir ningún frame de la carpeta.")

    fh, fw = first_frame.shape[:2]
    min_sep = MIN_ANCHOR_SEPARATION_FRAC * min(fh, fw)
    sep = float(np.hypot(anchor1[0] - anchor2[0], anchor1[1] - anchor2[1]))
    if sep < min_sep:
        raise ValueError(
            f"Los dos puntos están demasiado cerca ({sep:.0f}px < {min_sep:.0f}px). "
            "Sepáralos más para que haya suficiente base para detectar rotación."
        )

    template1, anchor1_in_tpl = _build_perforation_template(first_frame, anchor1)
    template2, anchor2_in_tpl = _build_perforation_template(first_frame, anchor2)
    if template1 is None:
        raise RuntimeError(
            f"No pude extraer plantilla 1 en ({anchor1[0]:.0f}, {anchor1[1]:.0f}). "
            "Selecciona un punto con más contraste."
        )
    if template2 is None:
        raise RuntimeError(
            f"No pude extraer plantilla 2 en ({anchor2[0]:.0f}, {anchor2[1]:.0f}). "
            "Selecciona un punto con más contraste."
        )

    log(
        f"Plantillas: 1={template1.shape[1]}×{template1.shape[0]} en "
        f"({anchor1[0]:.0f},{anchor1[1]:.0f}); 2={template2.shape[1]}×{template2.shape[0]} "
        f"en ({anchor2[0]:.0f},{anchor2[1]:.0f}); separación={sep:.0f}px."
    )

    perf_spacing = _detect_perf_spacing(first_frame, anchor1)
    if perf_spacing is None:
        raise RuntimeError(
            "No pude detectar el espaciado entre perforaciones en el primer frame. "
            "Asegúrate de que el primer punto esté sobre una perforación clara "
            "y que el encuadre incluya al menos tres perforaciones visibles."
        )
    log(f"Espaciado entre perforaciones: {perf_spacing:.0f}px.")

    search_radius = int(min(max(perf_spacing * 0.45, 80), perf_spacing - 40))
    predictor1 = _MotionPredictor(initial_pos=anchor1, alpha=0.6)
    predictor2 = _MotionPredictor(initial_pos=anchor2, alpha=0.6)

    # ── Phase 2: Per-anchor tracking passes ──
    log("Fase 2a: rastreando anclaje 1...")
    result1 = _track_anchor_pass(
        files,
        template1,
        anchor1_in_tpl,
        predictor1,
        search_radius,
        perf_spacing,
        log,
        "a1",
    )
    log("Fase 2b: rastreando anclaje 2...")
    result2 = _track_anchor_pass(
        files,
        template2,
        anchor2_in_tpl,
        predictor2,
        search_radius,
        perf_spacing,
        log,
        "a2",
    )

    # Bind to local names matching the prior tuple shape so downstream
    # logic (debug pass, summary, smoothing inputs) reads naturally.
    # Unit 5 will introduce result1.consensus_rejected / cons1; Unit 8
    # will use the same masks for the R13 health-check denominator.
    pos1, ncc1, amb1, rej1, fail1, out1 = (
        result1.positions,
        result1.nccs,
        result1.ambiguous_mask,
        result1.motion_rejected,
        result1.failed_mask,
        result1.outcomes,
    )
    pos2, ncc2, amb2, rej2, fail2, out2 = (
        result2.positions,
        result2.nccs,
        result2.ambiguous_mask,
        result2.motion_rejected,
        result2.failed_mask,
        result2.outcomes,
    )

    # Debug frames: save when either anchor failed on this frame.
    if debug_dir:
        for i, f in enumerate(files):
            a1_bad = pos1[i] is None
            a2_bad = pos2[i] is None
            if not (a1_bad or a2_bad):
                continue
            frame = cv2.imread(f)
            if frame is None:
                continue
            try:
                basename = os.path.splitext(os.path.basename(f))[0] + "_debug.jpg"
                debug_path = os.path.join(debug_dir, basename)
                dbg = frame.copy()
                for label_color, outcome in (
                    ((0, 255, 0), out1[i]),
                    ((0, 200, 255), out2[i]),
                ):
                    if outcome is None:
                        continue
                    for rank_idx, c in enumerate(outcome.ranked[:5]):
                        cx_c, cy_c, ncc_c, _sc = c
                        color = label_color if rank_idx == 0 else (0, 165, 255)
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
                    px_p, py_p = outcome.predicted
                    cv2.drawMarker(
                        dbg,
                        (int(px_p), int(py_p)),
                        (255, 0, 255),
                        cv2.MARKER_CROSS,
                        30,
                        2,
                    )
                cv2.imwrite(debug_path, dbg, [cv2.IMWRITE_JPEG_QUALITY, 80])
            except Exception as exc:
                log(f"Debug image failed: {exc}")

    both_detected = [
        (p1 is not None and p2 is not None) for p1, p2 in zip(pos1, pos2, strict=True)
    ]
    detected_count = sum(both_detected)
    failed_both_required = total - detected_count

    if progress_cb:
        progress_cb(0.5)

    if detected_count == 0:
        raise RuntimeError("No logré detectar ambos anclajes en ningún frame.")

    log(f"Detección: {detected_count}/{total} frames con ambos anclajes.")

    # ── Phase 3: Build raw transform series, smooth, then warp ──
    ref_pts = (tuple(map(float, anchor1)), tuple(map(float, anchor2)))

    tx_raw = np.full(total, np.nan, dtype=np.float64)
    ty_raw = np.full(total, np.nan, dtype=np.float64)
    theta_raw = np.full(total, np.nan, dtype=np.float64)
    for i, ok in enumerate(both_detected):
        if not ok:
            continue
        try:
            tx, ty, th = rigid_fit_2pt(ref_pts, (pos1[i], pos2[i]))
        except ValueError:
            continue
        tx_raw[i] = tx
        ty_raw[i] = ty
        theta_raw[i] = th

    splices = detect_splices(ncc1, ncc2, tx_raw, ty_raw, theta_raw, perf_spacing)
    tx_s, ty_s, theta_s, outlier_mask = smooth_trajectory(
        tx_raw,
        ty_raw,
        theta_raw,
        splices,
    )
    outlier_count = int(outlier_mask.sum())

    log(
        f"Suavizado: {len(splices)} segmento(s); {outlier_count} outlier(s) reemplazados."
    )

    # ── Phase 4: Warp each frame with its smoothed rigid transform ──
    log("Fase 3: aplicando transformaciones y guardando...")

    cv_border = _BORDER_MODES.get(border_mode, cv2.BORDER_REPLICATE)
    out_w = out_h = None

    for i, f in enumerate(files, 1):
        frame = cv2.imread(f)
        if frame is None:
            log(f"No pude abrir en la pasada de warp: {os.path.basename(f)}")
        else:
            h, w = frame.shape[:2]
            if out_h is None:
                out_h, out_w = h, w
            idx = i - 1
            # Translation-only warp. theta_s is computed by rigid_fit_2pt and
            # consumed by detect_splices as a discontinuity signal, but it is
            # intentionally NOT applied to the warp output here. Completes the
            # intent of commit 0c0c330 ("translation-only pipeline"). R4b's
            # absolute-position drift gate depends on this — see plan
            # docs/plans/2026-04-23-001-feat-stabilization-quality-pass-plan.md.
            c, s = 1.0, 0.0
            tx = float(tx_s[idx])
            ty = float(ty_s[idx])
            inv_tx = -tx
            inv_ty = -ty
            M = np.float32([[c, s, inv_tx], [-s, c, inv_ty]])
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
            progress_cb(0.5 + 0.5 * (i / total))

    summary = {
        "total_frames": total,
        "detected_both_frames": detected_count,
        "failed_frames_both_required": failed_both_required,
        "ambiguous_frames_a1": int(amb1.sum()),
        "ambiguous_frames_a2": int(amb2.sum()),
        "motion_rejected_frames_a1": int(rej1.sum()),
        "motion_rejected_frames_a2": int(rej2.sum()),
        "failed_detections_a1": int(fail1.sum()),
        "failed_detections_a2": int(fail2.sum()),
        "outlier_frames_replaced": outlier_count,
        "splice_count": max(0, len(splices) - 1),
        "splice_indices": [int(s) for s in splices[1:]],
        "perf_spacing_px": round(float(perf_spacing), 1),
        "search_radius_px": int(search_radius),
        "anchor1_ref": [round(float(anchor1[0]), 3), round(float(anchor1[1]), 3)],
        "anchor2_ref": [round(float(anchor2[0]), 3), round(float(anchor2[1]), 3)],
        "anchor_separation_px": round(sep, 1),
        "target_x": round(float(anchor1[0]), 3),
        "target_y": round(float(anchor1[1]), 3),
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
    log(f"Detectados (ambos anclajes): {detected_count}")
    log(f"Splices detectados: {summary['splice_count']}")
    log(f"Tamaño de salida: {out_w}×{out_h} px")
    return summary
