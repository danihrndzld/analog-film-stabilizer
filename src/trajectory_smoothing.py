"""Global trajectory smoothing + splice-aware segmentation.

Companion module to ``perforation_stabilizer_app.py``. Takes the raw
per-frame rigid transforms produced by the two-anchor tracker and returns
a gap-free, outlier-rejected, smoothed series ready for warp. Also exposes
the closed-form 2-point rigid fit used to derive per-frame transforms from
two matched anchor pairs.

Design assumptions (see
``docs/plans/2026-04-19-002-feat-multi-anchor-and-global-trajectory-smoothing-plan.md``):

- Exactly two anchors. Rigid fit is closed-form, no SVD.
- NaN marks a frame where at least one anchor failed; smoothing fills gaps.
- Outliers are flagged per-axis via MAD on residuals from an initial fit,
  then replaced and refit once.
- Splices segment the smoothing window so noise on one side does not leak
  across a discontinuity.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

DEFAULT_SAVGOL_WINDOW = 51
DEFAULT_SAVGOL_POLYORDER = 3
DEFAULT_MAD_MULTIPLIER = 3.5

DEFAULT_SPLICE_NCC_THRESHOLD = 0.25
DEFAULT_SPLICE_JUMP_MULT = 2.0
DEFAULT_SPLICE_ROT_JUMP_RAD = np.deg2rad(2.0)
DEFAULT_MIN_SEGMENT = 30


# ── Rigid fit ────────────────────────────────────────────────────────────────


def rigid_fit_2pt(ref_pts, obs_pts):
    """Closed-form rigid transform mapping ``ref_pts`` onto ``obs_pts``.

    Given two reference points ``R = (r1, r2)`` (positions on the first
    frame) and two observed points ``O = (o1, o2)`` (positions on a later
    frame), find the rigid transform ``T(p) = R_θ · p + t`` such that
    ``T(r_i) ≈ o_i``. With N=2 the solution is closed-form:

    - Rotation from the angle between ``r2 - r1`` and ``o2 - o1``.
    - Translation from the centroid delta after rotation.

    Parameters
    ----------
    ref_pts, obs_pts : sequence of two (x, y) pairs

    Returns
    -------
    (tx, ty, theta) : tuple of floats
        Translation in pixels, rotation in radians.

    Raises
    ------
    ValueError
        If either pair has zero separation (coincident points).
    """
    r = np.asarray(ref_pts, dtype=np.float64).reshape(2, 2)
    o = np.asarray(obs_pts, dtype=np.float64).reshape(2, 2)

    rv = r[1] - r[0]
    ov = o[1] - o[0]

    if float(np.hypot(*rv)) < 1e-6 or float(np.hypot(*ov)) < 1e-6:
        raise ValueError("rigid_fit_2pt: zero-separation anchor pair")

    # Angle from ref vector to observed vector: atan2(cross, dot).
    cross = rv[0] * ov[1] - rv[1] * ov[0]
    dot = rv[0] * ov[0] + rv[1] * ov[1]
    theta = float(np.arctan2(cross, dot))

    c, s = np.cos(theta), np.sin(theta)
    # Translation: observed_centroid - R · ref_centroid
    r_c = r.mean(axis=0)
    o_c = o.mean(axis=0)
    rotated_ref = np.array([c * r_c[0] - s * r_c[1], s * r_c[0] + c * r_c[1]])
    t = o_c - rotated_ref

    return float(t[0]), float(t[1]), theta


# ── Splice detection ─────────────────────────────────────────────────────────


def detect_splices(
    ncc_a1,
    ncc_a2,
    tx,
    ty,
    theta,
    perf_spacing,
    *,
    ncc_threshold=DEFAULT_SPLICE_NCC_THRESHOLD,
    jump_mult=DEFAULT_SPLICE_JUMP_MULT,
    rot_jump_rad=DEFAULT_SPLICE_ROT_JUMP_RAD,
    min_segment=DEFAULT_MIN_SEGMENT,
):
    """Detect splice / discontinuity frame indices.

    A splice boundary is declared at frame ``i`` when either:
      a) Both anchors collapsed NCC on that frame
         (``ncc_a1[i] < ncc_threshold`` AND ``ncc_a2[i] < ncc_threshold``), OR
      b) The inter-frame transform jump exceeds ``jump_mult × perf_spacing``
         on tx or ty, or ``rot_jump_rad`` on theta.

    NaN entries in ``tx/ty/theta`` are ignored for the jump signal (a failed
    frame is not a splice; it's just a detection gap). NCC arrays may
    contain NaN where the anchor failed; those contribute to the NCC-collapse
    signal as "failed" (below threshold).

    Returns
    -------
    list[int]
        Sorted list of frame indices at which a new segment starts. Always
        includes 0. Segments shorter than ``min_segment`` are merged back
        into the previous segment.
    """
    n = len(tx)
    if n == 0:
        return [0]

    ncc_a1 = np.asarray(ncc_a1, dtype=np.float64)
    ncc_a2 = np.asarray(ncc_a2, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    boundaries = {0}

    # (a) Dual NCC collapse. NaN counts as collapsed.
    bad_ncc = (np.nan_to_num(ncc_a1, nan=-1.0) < ncc_threshold) & (
        np.nan_to_num(ncc_a2, nan=-1.0) < ncc_threshold
    )
    for i in np.where(bad_ncc)[0]:
        if i > 0:
            boundaries.add(int(i))

    # (b) Inter-frame jump. Skip indices where either neighbour is NaN.
    jump_thresh = jump_mult * float(perf_spacing)
    for i in range(1, n):
        if (
            not (np.isfinite(tx[i]) and np.isfinite(tx[i - 1]))
            or not (np.isfinite(ty[i]) and np.isfinite(ty[i - 1]))
            or not (np.isfinite(theta[i]) and np.isfinite(theta[i - 1]))
        ):
            continue
        if (
            abs(tx[i] - tx[i - 1]) > jump_thresh
            or abs(ty[i] - ty[i - 1]) > jump_thresh
            or abs(theta[i] - theta[i - 1]) > rot_jump_rad
        ):
            boundaries.add(i)

    # Enforce min segment length. Sweep left-to-right; drop boundaries that
    # would create a segment shorter than `min_segment`.
    sorted_b = sorted(boundaries)
    kept = [sorted_b[0]]
    for b in sorted_b[1:]:
        if b - kept[-1] < min_segment:
            continue
        kept.append(b)
    # Also check final segment length; if it's too short, drop the last
    # boundary so the tail merges into the previous segment.
    if len(kept) > 1 and n - kept[-1] < min_segment:
        kept.pop()
    return kept


# ── Smoothing ────────────────────────────────────────────────────────────────


def _interp_nans(a):
    """Return a copy of ``a`` with NaN entries linearly interpolated."""
    a = np.asarray(a, dtype=np.float64).copy()
    good = np.isfinite(a)
    if good.all() or not good.any():
        return a
    idx = np.arange(len(a))
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def _savgol_segment(series, window, polyorder):
    """Run savgol on a 1D segment, shrinking the window if it's too long."""
    n = len(series)
    if n == 0:
        return series
    if n < polyorder + 2:
        # Degenerate segment: return median-filled.
        return np.full(n, float(np.median(series)))
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2:
        # Even small segments get polyorder+2 (or median fallback).
        w = polyorder + 2
        if w >= n:
            return np.full(n, float(np.median(series)))
        if w % 2 == 0:
            w += 1
    return savgol_filter(series, window_length=w, polyorder=polyorder, mode="nearest")


def smooth_trajectory(
    tx,
    ty,
    theta,
    splices,
    *,
    window=DEFAULT_SAVGOL_WINDOW,
    polyorder=DEFAULT_SAVGOL_POLYORDER,
    mad_multiplier=DEFAULT_MAD_MULTIPLIER,
):
    """Smooth three trajectory series segment-by-segment with MAD outlier refit.

    For each segment ``[splices[i], splices[i+1])``:
      1. Interpolate NaN gaps linearly.
      2. Initial savgol fit.
      3. Flag residual outliers per series where ``|res| > mad_multiplier ·
         MAD(res)``.
      4. Replace flagged samples with the initial fit value and refit savgol.

    Returns
    -------
    (tx_s, ty_s, theta_s, outlier_mask) : tuple of np.ndarray
        First three are the smoothed series (gap-free, same length as
        inputs). ``outlier_mask`` is a boolean array of length N flagging
        frames replaced as outliers on any of the three axes.
    """
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    n = len(tx)
    outlier_mask = np.zeros(n, dtype=bool)
    out_tx = np.empty(n, dtype=np.float64)
    out_ty = np.empty(n, dtype=np.float64)
    out_theta = np.empty(n, dtype=np.float64)

    if n == 0:
        return out_tx, out_ty, out_theta, outlier_mask

    bounds = list(splices) + [n]
    for seg_start, seg_end in zip(bounds[:-1], bounds[1:], strict=True):
        if seg_end <= seg_start:
            continue
        for src, dst in (
            (tx, out_tx),
            (ty, out_ty),
            (theta, out_theta),
        ):
            seg = src[seg_start:seg_end]
            filled = _interp_nans(seg)
            fit = _savgol_segment(filled, window, polyorder)
            residuals = filled - fit
            mad = float(np.median(np.abs(residuals - np.median(residuals))))
            if mad > 1e-9:
                # 1.4826 scales MAD to approximate sigma under Gaussian noise.
                threshold = mad_multiplier * 1.4826 * mad
                bad = np.abs(residuals) > threshold
                if bad.any():
                    outlier_mask[seg_start:seg_end] |= bad
                    # Replace outliers with NaN and re-interpolate so the
                    # refit is not contaminated by the bad sample's value.
                    filled[bad] = np.nan
                    filled = _interp_nans(filled)
                    fit = _savgol_segment(filled, window, polyorder)
            dst[seg_start:seg_end] = fit

    return out_tx, out_ty, out_theta, outlier_mask
