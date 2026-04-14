import glob
import os

import cv2
import numpy as np

VALID_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")

# Minimum contour area as a fraction of ROI area.  Calibrated so that at
# ~3000×2000 frames with roi_ratio=0.22 the effective floor ≈ 5 280 px,
# close to the previous hard-coded 5 000.
_MIN_AREA_FRAC = 0.004
_MIN_AREA_FLOOR = 200  # absolute floor to reject noise at very low resolutions


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


def _best_contour(
    thresh,
    roi_w,
    top_n=1,
    aspect_min=0.40,
    fill_min=0.75,
    solidity_min=0.85,
    collect_rejections=False,
):
    """Return top_n perforation candidates from a binary image as (cx, cy) tuples.

    The anchor point is the contour centroid (center of mass), which averages
    over the entire contour area.  This is far less sensitive to boundary noise
    than a corner point — a 1px boundary shift moves the centroid by ~1/N pixels
    (where N is the number of contour pixels).

    Parameters
    ----------
    thresh : np.ndarray
        Binary image (uint8) to search for contours.
    roi_w : int
        Width of the ROI strip; used for the x-position filter.
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

    roi_h = thresh.shape[0]
    min_area = max(_MIN_AREA_FLOOR, int(roi_h * roi_w * _MIN_AREA_FRAC))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)

        if area < min_area:
            if collect_rejections:
                rejections.append((x, y, bw, bh, "area"))
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        if solidity < solidity_min:
            if collect_rejections:
                rejections.append((x, y, bw, bh, "solidity"))
            continue

        # Compute centroid (center of mass) from contour moments.
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else float(x + bw / 2)
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else float(y + bh / 2)

        # X-position filter: reject perforations too far right in the ROI.
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

        # Axis-alignment filter: perforations are punched perpendicular to the
        # film strip, so the minAreaRect angle should be within ±8° of 0° or
        # 90°.  Diagonal scratch artefacts that pass area/solidity/fill checks
        # are typically rotated ~45° and are eliminated here.
        rect_angle = cv2.minAreaRect(cnt)[2]  # degrees in [-90, 0)
        if abs(rect_angle) > 8 and abs(rect_angle + 90) > 8:
            if collect_rejections:
                rejections.append((x, y, bw, bh, "angle"))
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
        font_scale_name = max(0.4, h / 2500.0)
        thickness = 1

        for rx, ry, rbw, rbh, reason in rejections:
            cv2.rectangle(img, (rx, ry), (rx + rbw, ry + rbh), (0, 0, 255), 2)
            label_y = max(ry - 4, int(font_scale_reason * 20))
            cv2.putText(
                img,
                reason,
                (rx, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_reason,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )

        # Filename overlay in yellow at the top-left
        overlay_y = max(16, int(h * 0.015))
        cv2.putText(
            img,
            frame_name,
            (4, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_name,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        basename = os.path.splitext(frame_name)[0] + "_debug.jpg"
        out_path = os.path.join(debug_dir, basename)
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    except Exception:
        # Never let a debug-image write error abort the stabilisation run.
        pass


def _annotate_roi_preview(roi_bgr, anchor, rejections, frame_name="", roi_boundary_x=None):
    """Return an annotated copy of a preview image for the UI preview panel.

    When anchor is not None (detection succeeded), draws a green crosshair and
    a "DETECTADO" label at the anchor position.  When anchor is None, draws
    rejected contour bounding rects in red with reason labels and a "NO
    DETECTADO" overlay — mirroring the _save_debug_frame style.

    When roi_boundary_x is provided, draws a subtle vertical line to indicate
    the detection zone boundary (the preview may be wider than the ROI to
    provide context).

    Does NOT write to disk; the caller is responsible for saving.

    Parameters
    ----------
    roi_bgr : np.ndarray
        BGR preview crop (may be wider than ROI for context).
    anchor : tuple(float, float) or None
        Detected anchor (cx, cy) in full-frame coordinates, or None.
    rejections : list of (x, y, bw, bh, reason_str)
        Rejected contour info from _best_contour(collect_rejections=True).
    frame_name : str
        Source frame basename used for the filename overlay label.
    roi_boundary_x : int or None
        X coordinate of the ROI boundary. When provided, draws a vertical line
        to show where detection happens vs. context area.

    Returns
    -------
    np.ndarray  annotated BGR image with same shape as roi_bgr.
    """
    img = roi_bgr.copy()
    h, w = img.shape[:2]

    font_scale = max(0.3, h / 3000.0)
    thickness = 1

    # ROI boundary indicator (dashed amber line)
    if roi_boundary_x is not None and 0 < roi_boundary_x < w:
        amber = (0, 191, 255)  # BGR for amber/orange
        dash_len = max(8, int(h * 0.01))
        gap_len = max(4, int(h * 0.005))
        y = 0
        while y < h:
            y_end = min(y + dash_len, h)
            cv2.line(img, (roi_boundary_x, y), (roi_boundary_x, y_end), amber, 1, cv2.LINE_AA)
            y = y_end + gap_len

    # Frame-name overlay (top-left, yellow)
    if frame_name:
        overlay_y = max(16, int(h * 0.015))
        cv2.putText(
            img,
            frame_name,
            (4, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, h / 2500.0),
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    if anchor is not None and np.isfinite(anchor[0]) and np.isfinite(anchor[1]):
        cx, cy = int(round(anchor[0])), int(round(anchor[1]))
        color_found = (0, 220, 0)  # green

        # Crosshair
        cv2.line(img, (0, cy), (w, cy), color_found, 1, cv2.LINE_AA)
        cv2.line(img, (cx, 0), (cx, h), color_found, 1, cv2.LINE_AA)

        # Small filled circle at intersection
        cv2.circle(img, (cx, cy), max(4, int(h * 0.005)), color_found, -1, cv2.LINE_AA)

        # Label
        label_y = max(cy - 8, int(font_scale * 20))
        cv2.putText(
            img,
            "DETECTADO",
            (max(0, cx + 6), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color_found,
            thickness,
            cv2.LINE_AA,
        )
    else:
        # Draw rejection boxes in red
        for rx, ry, rbw, rbh, reason in rejections:
            cv2.rectangle(img, (rx, ry), (rx + rbw, ry + rbh), (0, 0, 255), 2)
            label_y = max(ry - 4, int(font_scale * 20))
            cv2.putText(
                img,
                reason,
                (rx, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )

        # "NO DETECTADO" banner
        banner_y = min(h - 8, int(h * 0.95))
        cv2.putText(
            img,
            "NO DETECTADO",
            (4, banner_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, h / 2000.0),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def _detect_by_edge_projection(
    roi_bgr,
    frame_h,
    film_format,
    roi_w,
    edge_gauss_sigma=15.0,
    edge_peak_thresh_hi=0.60,
    edge_peak_thresh_lo=0.15,
):
    """Detect a perforation using horizontal Sobel gradient projection.

    Applies Sobel-Y to the ROI to detect horizontal brightness transitions,
    collapses the gradient to a 1-D row-mean profile, Gaussian-smooths it,
    then finds a pair of local maxima whose spacing matches the expected
    perforation height.  The midpoint of that pair is the centroid.

    This method detects the perf boundary from gradient *transitions* rather
    than requiring a closed, well-thresholded blob.  An 8mm Regular
    perforation produces two gradient peaks: one at its top edge and one at
    its bottom edge (~22-24 % of frame height apart at Diego's scanner
    resolution).  Overexposed frames where Otsu finds no bimodal boundary
    still retain these gradient steps, which is why this method recovers
    failures that the contour path cannot.

    Parameters
    ----------
    roi_bgr : np.ndarray
        BGR ROI crop (left strip of the source frame, full frame height).
    frame_h : int
        Full frame height in pixels; used for the zone guard.
    film_format : str
        Film format string ('super8', '8mm', 'super16').  The vertical zone
        guard (cy < 30 % of frame height) is applied only for '8mm' and
        'super16'.
    roi_w : int
        Width of the ROI strip; used to derive cx = roi_w / 2.
    edge_gauss_sigma : float
        Gaussian smoothing σ (in rows) applied to the row-mean profile.
        Default 15.0 smooths over ~45 rows at 3σ, suppressing grain noise
        while preserving the broad perf-boundary peaks.
    edge_peak_thresh_hi : float
        Minimum peak height as a fraction of profile.max() for a local
        maximum to be considered a candidate perf boundary.  Default 0.60.
    edge_peak_thresh_lo : float
        Minimum perf height as a fraction of ROI height.  Peak pairs closer
        together than this fraction are rejected as spurious (dust, grain
        blobs, narrow scratches).  Default 0.15 (≈456 px at 3040 px frame
        height), safely above the ~4 % spurious pairs and well below the
        ~22 % true perf span.

    Returns
    -------
    (float, float) or None
        (cx, cy) of the detected perforation centroid, or None if not found.
        cx is nan (interpolated from neighbors); cy is the midpoint of the perf.
    """
    # 1. Grayscale + Sobel-Y (horizontal edge detection)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    profile = np.mean(np.abs(sobel), axis=1)  # shape: (roi_h,)

    roi_h = roi_bgr.shape[0]

    # 2. Guard: all-dark / all-uniform ROI → no useful gradient
    if profile.max() < 1.0:
        return None

    # 3. Gaussian smooth via np.convolve (no scipy dependency).
    #    Build a kernel covering ±3σ; ensure odd length.
    sigma = edge_gauss_sigma
    half_w = max(1, int(3 * sigma))
    x = np.arange(-half_w, half_w + 1, dtype=np.float64)
    gauss_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gauss_kernel /= gauss_kernel.sum()
    profile_smooth = np.convolve(
        np.pad(profile, (half_w, half_w), mode="edge"),
        gauss_kernel,
        mode="valid",
    )

    # 4. Guard: all-uniform after smoothing
    p_max = profile_smooth.max()
    if p_max < 1.0:
        return None

    # 5. Find local maxima that exceed edge_peak_thresh_hi × p_max.
    #    A row qualifies as a local maximum when it is the highest point
    #    within ±min_sep rows (at least 3 % of ROI height, minimum 10 rows).
    threshold = edge_peak_thresh_hi * p_max
    min_sep = max(10, int(roi_h * 0.03))
    peaks = []
    for r in range(min_sep, roi_h - min_sep):
        lo, hi = r - min_sep, r + min_sep
        if (
            profile_smooth[r] >= profile_smooth[lo : hi + 1].max()
            and profile_smooth[r] >= threshold
        ):
            peaks.append(r)

    if not peaks:
        return None

    # 6. For multi-perf formats: find the topmost pair (r_top, r_bot) where
    #    the pair spans ≥ edge_peak_thresh_lo of ROI height (rejects spurious
    #    narrow pairs) and ≤ 35 % (rejects full-frame false pairs), and whose
    #    midpoint cy falls within the top-perf zone (< 30 % of frame height).
    #    Take the first valid pair found (topmost).
    for i, r_top in enumerate(peaks):
        for r_bot in peaks[i + 1 :]:
            perf_h = r_bot - r_top
            if perf_h < edge_peak_thresh_lo * roi_h or perf_h > 0.35 * roi_h:
                continue
            cy = (r_top + r_bot) / 2.0
            if film_format in ("8mm", "super16") and cy >= frame_h * 0.30:
                continue
            return (float("nan"), float(cy))

    # 7. Super 8 fallback: single centred perf — use topmost peak directly.
    #    Zone guard does not apply for super8.
    if film_format == "super8":
        return (float("nan"), float(peaks[0]))

    return None


def detect_perforation(
    frame,
    roi_ratio=0.22,
    threshold=210,
    film_format="super8",
    debug_dir=None,
    frame_name="",
    edge_gauss_sigma=15.0,
    edge_peak_thresh_hi=0.60,
    edge_peak_thresh_lo=0.15,
):
    """Detect the perforation anchor point in a frame.

    Detection uses a two-stage cascade:

      1. Contour detection (primary) — CLAHE + Otsu + adaptive threshold +
         contour filtering.  The contour centroid averages over the full
         perforation-hole area, producing stable frame-to-frame measurements.

      2. Edge projection (fallback) — Sobel-Y gradient projected to a 1-D
         row-mean profile.  Runs only when contour detection fails (typically
         overexposed frames where Otsu cannot produce a closed blob).  Detects
         the perf boundary from brightness transitions rather than blob shape.

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

    Edge projection tuning parameters (code-level only, not exposed in UI/CLI):
      edge_gauss_sigma      — Gaussian smoothing σ for the row-mean profile.
      edge_peak_thresh_hi   — Top-edge threshold as fraction of profile max.
      edge_peak_thresh_lo   — Bottom-edge threshold as fraction of profile max.
    """
    h, w = frame.shape[:2]
    roi_w = max(50, int(w * roi_ratio))
    kernel = np.ones((3, 3), np.uint8)

    def thresh_band(band_bgr):
        """Threshold a band with a three-tier fallback strategy.

        Tier 1 — White-reference adaptive threshold:
          Sample the right 15 % of the ROI (film base, never the perf) to
          measure the actual film-base brightness, then threshold at 85 % of
          that value.  This normalises for per-frame exposure drift and is
          the primary fix for overexposed 8mm Regular frames where Otsu finds
          no bimodal boundary.

        Tier 2 — Otsu:
          Used when the film base is too dark for the reference strip to give
          a reliable reading, or when Tier 1 produces a saturated result.

        Tier 3 — Adaptive threshold:
          Fallback when Otsu produces an all-black OR near-saturated result.
        """
        gray_raw = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
        bh_b, bw_b = band_bgr.shape[:2]

        # ── Tier 1: white-reference threshold ─────────────────────────────
        ref_x = int(bw_b * 0.85)
        if ref_x < bw_b:
            ref_mean = float(np.mean(gray_raw[:, ref_x:]))
            # Only use white-reference threshold when the film base is genuinely
            # bright (overexposed).  At ref_mean ≤ 190 the film is normally or
            # under-exposed; Otsu handles it well and a low reference threshold
            # (ref_mean × 0.85 ≈ 85–160) would be too permissive, flooding the
            # binary and masking the perforation.
            if ref_mean > 190:
                blur_raw = cv2.GaussianBlur(gray_raw, (5, 5), 0)
                _, t_ref = cv2.threshold(
                    blur_raw, ref_mean * 0.85, 255, cv2.THRESH_BINARY
                )
                t_ref = cv2.morphologyEx(t_ref, cv2.MORPH_OPEN, kernel)
                t_ref = cv2.morphologyEx(t_ref, cv2.MORPH_CLOSE, kernel)
                nz_ref = cv2.countNonZero(t_ref)
                if 0 < nz_ref < int(0.95 * t_ref.size):
                    return t_ref

        # ── Tier 2: CLAHE + Otsu ──────────────────────────────────────────
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray_raw)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, t = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel)
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)
        nz = cv2.countNonZero(t)
        if 0 < nz < int(0.95 * t.size):
            return t

        # ── Tier 3: adaptive threshold ────────────────────────────────────
        block = max(51, (min(bh_b, roi_w) // 20) | 1)
        a = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -5
        )
        a = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
        a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
        return a

    roi = frame[:, :roi_w]

    if film_format in ("8mm", "super16"):
        # ── Primary: contour detection ─────────────────────────────────────────
        # Single-pass full-ROI scan — no h//2 split.  The contour centroid
        # averages over the full perf-hole area; it is considerably more stable
        # frame-to-frame than the gradient peak midpoint produced by edge
        # projection, which is why contour runs first.
        binary = thresh_band(roi)

        if debug_dir:
            candidates, rejections = _best_contour(
                binary,
                roi_w,
                top_n=2,
                aspect_min=0.25,
                fill_min=0.65,
                solidity_min=0.80,
                collect_rejections=True,
            )
        else:
            candidates = _best_contour(
                binary,
                roi_w,
                top_n=2,
                aspect_min=0.25,
                fill_min=0.65,
                solidity_min=0.80,
            )
            rejections = []

        if len(candidates) >= 1:
            # Per-hole quality validation: for 8mm/super16 the top perforation
            # sits at the frame boundary — its anchor (bottom-right corner)
            # should appear in the upper 35 % of the frame height.  Candidates
            # below that threshold are almost certainly the bottom perf, a
            # frame-separator artefact, or bright film content leaking into
            # the ROI.
            #
            # Keeping only top-zone candidates (centroid cy < 30 % of frame):
            #   • Eliminates the false-positive zone between the two perfs.
            #   • Recovers frames where one valid top-perf contour was found
            #     alongside a spurious bottom candidate — previously the
            #     midpoint would be corrupted; now we use the top perf alone.
            #   • Replaces the old "reject if single candidate is in bottom
            #     half" rule with a tighter, format-aware band.
            top_zone = [p for p in candidates if p[1] < h * 0.30]
            if top_zone:
                pt_top = min(top_zone, key=lambda p: p[1])
                return (float(pt_top[0]), float(pt_top[1]))

        # ── Fallback: edge projection ──────────────────────────────────────────
        # Runs only when contour detection found no valid candidate in the top
        # zone.  Edge projection detects the perf boundary from Sobel gradient
        # transitions — it recovers overexposed frames where Otsu fails to
        # produce a closed blob.  collect_rejections is set only here (contour
        # already failed) to avoid computing rejection lists on successful frames.
        ep_result = _detect_by_edge_projection(
            roi,
            h,
            film_format,
            roi_w,
            edge_gauss_sigma=edge_gauss_sigma,
            edge_peak_thresh_hi=edge_peak_thresh_hi,
            edge_peak_thresh_lo=edge_peak_thresh_lo,
        )
        if ep_result is not None:
            return ep_result

        # Both methods failed — save debug image if requested.
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

    # Fallback for Super 8 overexposed frames.
    ep_result = _detect_by_edge_projection(
        roi,
        h,
        film_format,
        roi_w,
        edge_gauss_sigma=edge_gauss_sigma,
        edge_peak_thresh_hi=edge_peak_thresh_hi,
        edge_peak_thresh_lo=edge_peak_thresh_lo,
    )
    if ep_result is not None:
        return ep_result

    if debug_dir and frame_name:
        _save_debug_frame(roi, rejections, frame_name, debug_dir)
    return None


_BORDER_MODES = {
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT_101,
}


# ── Template matching alignment ──────────────────────────────────────────────


def _build_perforation_template(frame, anchor, roi_ratio=0.22, padding=20):
    """Extract a grayscale patch around the detected perforation for template matching.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame.
    anchor : tuple(float, float)
        Detected (cx, cy) in full-frame coordinates.
    roi_ratio : float
        Fraction of frame width used as ROI.
    padding : int
        Extra pixels around the perforation bounding region.

    Returns
    -------
    (template, origin) : (np.ndarray, (int, int))
        Grayscale template patch and the (x0, y0) origin of the patch in the ROI.
        Returns (None, None) if extraction fails.
    """
    h, w = frame.shape[:2]
    roi_w = max(50, int(w * roi_ratio))
    roi_bgr = frame[:, :roi_w]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    cx, cy = anchor
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None, None
    # Estimate perforation size from ROI dimensions (typically ~8-15% of ROI width)
    est_perf_w = int(roi_w * 0.35)
    est_perf_h = int(h * 0.08)

    x0 = max(0, int(cx - est_perf_w // 2 - padding))
    y0 = max(0, int(cy - est_perf_h // 2 - padding))
    x1 = min(roi_w, int(cx + est_perf_w // 2 + padding))
    y1 = min(h, int(cy + est_perf_h // 2 + padding))

    if x1 - x0 < 10 or y1 - y0 < 10:
        return None, None

    template = gray[y0:y1, x0:x1].copy()
    return template, (x0, y0)


def _build_averaged_template(frames_and_anchors, roi_ratio=0.22, padding=20):
    """Build a noise-reduced template by averaging patches from multiple frames.

    Parameters
    ----------
    frames_and_anchors : list of (np.ndarray, (float, float))
        List of (frame, anchor) tuples from successful detections.
    roi_ratio : float
        Fraction of frame width used as ROI.
    padding : int
        Extra pixels around the perforation.

    Returns
    -------
    (template, origin) : (np.ndarray, (int, int)) or (None, None)
    """
    patches = []
    origin = None
    for frame, anchor in frames_and_anchors:
        tpl, orig = _build_perforation_template(frame, anchor, roi_ratio, padding)
        if tpl is not None:
            if origin is None:
                origin = orig
                patches.append(tpl.astype(np.float32))
            elif tpl.shape == patches[0].shape:
                patches.append(tpl.astype(np.float32))

    if not patches:
        return None, None

    averaged = np.mean(patches, axis=0).astype(np.uint8)
    return averaged, origin


def _template_match_perforation(frame, template, roi_ratio=0.22, min_confidence=0.4):
    """Locate the perforation in a frame using normalized cross-correlation.

    Uses cv2.matchTemplate with TM_CCOEFF_NORMED and parabolic sub-pixel
    refinement on the correlation peak.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame.
    template : np.ndarray
        Grayscale perforation template patch.
    roi_ratio : float
        Fraction of frame width used as ROI.
    min_confidence : float
        Minimum NCC score to accept the match (0.0-1.0).

    Returns
    -------
    (cx, cy, confidence) or None
        Sub-pixel (cx, cy) in ROI coordinates and NCC confidence score,
        or None if the match is below min_confidence.
    """
    h, w = frame.shape[:2]
    roi_w = max(50, int(w * roi_ratio))
    roi_bgr = frame[:, :roi_w]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

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


def _estimate_rotation(frame, template, cx, cy, roi_ratio=0.22):
    """Estimate small rotation angle of the perforation region using ECC.

    Uses cv2.findTransformECC with MOTION_EUCLIDEAN (translation + rotation)
    on a tight crop around the template-matched position.  Returns the rotation
    angle in degrees, or 0.0 if ECC fails to converge.

    Parameters
    ----------
    frame : np.ndarray
        Full BGR frame.
    template : np.ndarray
        Grayscale perforation template patch.
    cx, cy : float
        Template-matched center position in ROI coordinates.
    roi_ratio : float
        Fraction of frame width used as ROI.

    Returns
    -------
    float
        Rotation angle in degrees (positive = counter-clockwise).
    """
    h, w = frame.shape[:2]
    roi_w = max(50, int(w * roi_ratio))
    roi_bgr = frame[:, :roi_w]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    th, tw = template.shape[:2]
    # Extract the matched region from the current frame
    x0 = max(0, int(cx - tw / 2))
    y0 = max(0, int(cy - th / 2))
    x1 = min(roi_w, x0 + tw)
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
    progress_cb=None,
    log_cb=None,
    roi_ratio=0.22,
    threshold=210,
    smooth_radius=9,
    jpeg_quality=0,
    film_format="super8",
    debug_dir=None,
    manual_anchor=None,
    border_mode="replicate",
):
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

    # ── Phase 1: Bootstrap — build reference template from first detections ──
    log("Fase 1: construyendo plantilla de referencia...")
    bootstrap_limit = min(30, total)
    bootstrap_frames = []  # (frame, anchor) pairs for template building
    template = None

    for i, f in enumerate(files[:bootstrap_limit], 1):
        frame = cv2.imread(f)
        if frame is None:
            continue
        pt = detect_perforation(
            frame,
            roi_ratio=roi_ratio,
            threshold=threshold,
            film_format=film_format,
        )
        if pt is not None and np.isfinite(pt[0]) and np.isfinite(pt[1]):
            bootstrap_frames.append((frame, pt))
            if len(bootstrap_frames) >= 15:
                break

    if bootstrap_frames:
        template, _tpl_origin = _build_averaged_template(
            bootstrap_frames, roi_ratio=roi_ratio, padding=25
        )
        if template is not None:
            log(
                f"Plantilla construida de {len(bootstrap_frames)} frames "
                f"({template.shape[1]}×{template.shape[0]} px)"
            )

    if template is None:
        log("No pude construir plantilla — usando detección por contorno.")

    # ── Phase 2: Detect all frames via template matching (or contour fallback) ──
    log("Fase 2: alineando frames...")
    tm_count = 0
    contour_count = 0

    for i, f in enumerate(files, 1):
        frame = cv2.imread(f)
        if frame is None:
            points.append(None)
            failures += 1
            log(f"No pude abrir: {os.path.basename(f)}")
        else:
            pt = None

            angle = 0.0

            # Try template matching first (primary)
            if template is not None:
                tm_result = _template_match_perforation(
                    frame, template, roi_ratio=roi_ratio, min_confidence=0.4
                )
                if tm_result is not None:
                    cx, cy, conf = tm_result
                    pt = (cx, cy)
                    tm_count += 1
                    # Estimate rotation from the matched region
                    angle = _estimate_rotation(
                        frame, template, cx, cy, roi_ratio=roi_ratio
                    )

            # Fall back to contour detection
            if pt is None:
                pt = detect_perforation(
                    frame,
                    roi_ratio=roi_ratio,
                    threshold=threshold,
                    film_format=film_format,
                    debug_dir=debug_dir,
                    frame_name=os.path.basename(f),
                )
                if pt is not None:
                    contour_count += 1

            points.append(pt)
            angles.append(angle)
            if pt is None:
                failures += 1
                log(f"Sin detección: {os.path.basename(f)}")
        if progress_cb:
            progress_cb(i / (total * 2))

    log(
        f"Detección: {tm_count} template, {contour_count} contorno, "
        f"{failures} fallidos de {total}"
    )

    valid = [p for p in points if p is not None]
    if not valid:
        raise RuntimeError("No logré detectar la perforación en ningún frame.")

    # Compute robust target: use manual anchor when provided, otherwise median of detections.
    if manual_anchor is not None:
        target_x = float(manual_anchor[0])
        target_y = float(manual_anchor[1])
        log(f"Usando referencia manual: x={target_x:.2f}, y={target_y:.2f}")
    else:
        # IQR outlier rejection: drop positions outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR]
        # before computing the median.  This removes residual wrong-perf detections
        # (e.g. bottom-perf anchors that slipped past the position guard) without
        # affecting batches where all anchors are tightly clustered.
        ys_valid = np.array([p[1] for p in valid], dtype=np.float32)
        q1, q3 = float(np.percentile(ys_valid, 25)), float(np.percentile(ys_valid, 75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inliers = [p for p in valid if lo <= p[1] <= hi]
        if len(inliers) < len(valid):
            log(
                f"IQR: descartados {len(valid) - len(inliers)} anclas atípicas "
                f"(fuera de [{lo:.0f}, {hi:.0f}] px)"
            )
        if inliers:
            valid = inliers
        xs_for_iqr = np.array([p[0] for p in valid], dtype=np.float32)
        finite_cx = xs_for_iqr[np.isfinite(xs_for_iqr)]
        if len(finite_cx) == 0:
            raise RuntimeError("No se encontraron frames con detección de cx válida.")
        q1x = float(np.percentile(finite_cx, 25))
        q3x = float(np.percentile(finite_cx, 75))
        iqrx = q3x - q1x
        lox, hix = q1x - 1.5 * iqrx, q3x + 1.5 * iqrx
        xs_for_target = np.where(
            np.isfinite(xs_for_iqr) & ((xs_for_iqr < lox) | (xs_for_iqr > hix)),
            np.nan,
            xs_for_iqr,
        )
        x_rejected = int(np.sum(np.isfinite(xs_for_iqr) & ~np.isfinite(xs_for_target)))
        if x_rejected > 0:
            log(
                f"IQR X: descartados {x_rejected} anclas atípicas "
                f"(fuera de [{lox:.0f}, {hix:.0f}] px)"
            )
        target_x = float(np.nanmedian(xs_for_target))
        target_y = float(np.median([p[1] for p in valid]))
    log(f"Punto fijo objetivo: x={target_x:.2f}, y={target_y:.2f}")

    # Fill None positions (failed detections) by linear interpolation.
    xs = np.array([p[0] if p is not None else np.nan for p in points], dtype=np.float32)
    ys = np.array([p[1] if p is not None else np.nan for p in points], dtype=np.float32)
    # X IQR: null cx values outside the inlier band before interpolation fills them.
    # Use direct threshold on xs (length N); do NOT use a mask from xs_for_iqr
    # (length M = len(valid)) — arrays are not index-aligned.
    if manual_anchor is None:
        xs[(np.isfinite(xs)) & ((xs < lox) | (xs > hix))] = np.nan
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
            # Rotate around the detected perforation position to correct
            # in-gate tilt, then translate to align to the target.
            if abs(angle) > 0.001:
                # Rotate around the perforation's position in the frame
                perf_x, perf_y = pt[0], pt[1]
                R = cv2.getRotationMatrix2D((perf_x, perf_y), -angle, 1.0)
                # Add translation to the rotation matrix
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
        "film_format": film_format,
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
