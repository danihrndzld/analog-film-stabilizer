"""Phase 0 calibration.

Scans N frames sampled evenly across the batch to estimate perforation
spacing, per-anchor template shape, and a reference NCC distribution
*before* per-frame tracking begins. The output `CalibrationState` is
consumed by Pass 1 (consensus gates), Pass 2 (smoothed-prior rescue),
and the run summary.

See docs/plans/2026-04-23-001-feat-stabilization-quality-pass-plan.md
(Unit 1) for the full contract. Key invariants enforced here:

- `anchor1_ref` / `anchor2_ref` are the **user click** verbatim
  (Decision C5) — the median observed position is recorded separately
  as a diagnostic, not used as the gate baseline.
- ≥20 of `n_samples` (default 30) frames must yield usable spacing +
  templates + NCC values, else `run_calibration` returns `None` and the
  caller falls back to single-frame bootstrap (R2).
- I/O failures are skip-and-continue (logged), not retried.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from perforation_stabilizer_app import (
    _build_perforation_template,
    _recover_perf_spacing,
    _template_match_candidates,
)

# ── Phase 0 calibration thresholds and defaults ─────────────────────────────
DEFAULT_N_SAMPLES = 30
EFFECTIVE_N_FLOOR = 20
SPACING_STABILITY_RATIO = 0.10  # std/mean must stay below this
NCC_TOP_P50_FLOOR = 0.50
CLICK_OFF_PERF_FRACTION = 0.25  # warn when |click - median| > this · perf_spacing


@dataclass
class CalibrationState:
    """Phase 0 outputs consumed by Pass 1, R4 gates, Pass 2, and the summary."""

    perf_spacing: float
    template_a1: np.ndarray
    template_a2: np.ndarray
    anchor1_ref: tuple[float, float]
    anchor2_ref: tuple[float, float]
    ncc_top_p50: float
    ncc_top_p10: float
    ncc_runner_up_p50: float
    effective_n: int
    sampled_indices: list[int]
    observed_median_a1_x: float
    observed_median_a1_y: float
    observed_median_a2_x: float
    observed_median_a2_y: float


def _sample_indices(n_files: int, n_samples: int) -> list[int]:
    """Evenly-spaced sample indices, deduped while preserving order."""
    if n_files <= 0:
        return []
    raw = np.linspace(0, n_files - 1, n_samples).astype(int)
    seen: set[int] = set()
    out: list[int] = []
    for idx in raw:
        i = int(idx)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _measure_anchor_ncc(
    gray: np.ndarray,
    anchor: tuple[float, float],
    template: np.ndarray,
    anchor_in_tpl: tuple[float, float],
    search_radius: int,
):
    """Run top-2 NCC at one anchor on one frame.

    Returns dict with keys: top_ncc (float|None), runner_up_ncc (float|None),
    observed_x (float|None), observed_y (float|None). Any field can be None
    when the candidate list is empty.
    """
    out = {
        "top_ncc": None,
        "runner_up_ncc": None,
        "observed_x": None,
        "observed_y": None,
    }

    candidates = _template_match_candidates(
        gray,
        template,
        k=2,
        search_center=anchor,
        search_radius=search_radius,
        anchor_in_tpl=anchor_in_tpl,
    )
    if candidates:
        cx, cy, ncc = candidates[0]
        out["top_ncc"] = float(ncc)
        out["observed_x"] = float(cx)
        out["observed_y"] = float(cy)
        if len(candidates) >= 2:
            out["runner_up_ncc"] = float(candidates[1][2])

    return out


def run_calibration(
    files: list[str],
    anchor1: tuple[float, float],
    anchor2: tuple[float, float],
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    log: Callable[[str], None],
) -> CalibrationState | None:
    """Sample N frames evenly across the batch and return calibrated state.

    Returns ``None`` when calibration cannot stabilize (effective_n < 20,
    spacing std/mean ≥ 0.10, NCC top-p50 < 0.5, or an anchor that cannot
    be templated). The caller is expected to fall back to single-frame
    bootstrap with a "calibration-unverified" warning per R2.

    All log output goes through the supplied ``log(msg: str)`` callback so
    the Electron CLI can route messages over JSON-lines.
    """
    if not files:
        log("Calibración: no hay frames de entrada")
        return None

    indices = _sample_indices(len(files), n_samples)
    log(f"Calibrando: muestreando {len(indices)} frames de {len(files)}")

    # Build templates from the first readable sampled frame. We require both
    # anchors to template successfully on the same frame so the templates
    # share the same lighting / capture conditions.
    template_a1 = template_a2 = None
    anchor1_in_tpl = anchor2_in_tpl = None
    for i in indices:
        frame = cv2.imread(files[i])
        if frame is None:
            continue
        t1, ait1 = _build_perforation_template(frame, anchor1)
        t2, ait2 = _build_perforation_template(frame, anchor2)
        if t1 is not None and t2 is not None:
            template_a1, anchor1_in_tpl = t1, ait1
            template_a2, anchor2_in_tpl = t2, ait2
            break

    if template_a1 is None or template_a2 is None:
        log(
            "Calibración: no pude construir templates en ninguno de los "
            "frames muestreados (¿anchor fuera del frame?)"
        )
        return None

    spacings: list[float] = []
    ncc_top_a1: list[float] = []
    ncc_top_a2: list[float] = []
    ncc_runner_a1: list[float] = []
    ncc_runner_a2: list[float] = []
    obs_a1_x: list[float] = []
    obs_a1_y: list[float] = []
    obs_a2_x: list[float] = []
    obs_a2_y: list[float] = []
    effective_n = 0

    for i in indices:
        frame = cv2.imread(files[i])
        if frame is None:
            log(f"Calibración: no pude abrir frame {i} (saltando)")
            continue

        # Spacing is detected once per frame. The primary contour detector
        # needs ≥3 visible bright contours; when exposure defeats that, fall
        # back to the same templates that tracking will use.
        spacing = _recover_perf_spacing(
            frame,
            anchor1,
            anchor2,
            template1=template_a1,
            anchor1_in_tpl=anchor1_in_tpl,
            template2=template_a2,
            anchor2_in_tpl=anchor2_in_tpl,
        )
        if spacing is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ncc_search_r = int(max(spacing * 0.5, 80))
        m1 = _measure_anchor_ncc(
            gray, anchor1, template_a1, anchor1_in_tpl, ncc_search_r
        )
        m2 = _measure_anchor_ncc(
            gray, anchor2, template_a2, anchor2_in_tpl, ncc_search_r
        )

        if m1["top_ncc"] is None or m2["top_ncc"] is None:
            continue

        spacings.append(spacing)
        ncc_top_a1.append(m1["top_ncc"])
        ncc_top_a2.append(m2["top_ncc"])
        if m1["runner_up_ncc"] is not None:
            ncc_runner_a1.append(m1["runner_up_ncc"])
        if m2["runner_up_ncc"] is not None:
            ncc_runner_a2.append(m2["runner_up_ncc"])
        if m1["observed_x"] is not None:
            obs_a1_x.append(m1["observed_x"])
            obs_a1_y.append(m1["observed_y"])
        if m2["observed_x"] is not None:
            obs_a2_x.append(m2["observed_x"])
            obs_a2_y.append(m2["observed_y"])
        effective_n += 1

    if effective_n < EFFECTIVE_N_FLOOR:
        log(
            f"Calibración inestable: solo {effective_n} de {len(indices)} "
            f"frames muestreados produjeron mediciones útiles "
            f"(mínimo {EFFECTIVE_N_FLOOR})"
        )
        return None

    spacings_arr = np.asarray(spacings, dtype=np.float64)
    spacing_mean = float(spacings_arr.mean())
    spacing_std = float(spacings_arr.std())
    if spacing_mean <= 0:
        log("Calibración: spacing media ≤ 0, abortando")
        return None
    spacing_ratio = spacing_std / spacing_mean
    if spacing_ratio >= SPACING_STABILITY_RATIO:
        log(
            f"Calibración inestable: spacing std/mean = {spacing_ratio:.3f} "
            f"(máximo {SPACING_STABILITY_RATIO})"
        )
        return None
    perf_spacing = float(np.median(spacings_arr))

    ncc_combined = np.asarray(ncc_top_a1 + ncc_top_a2, dtype=np.float64)
    ncc_top_p50 = float(np.median(ncc_combined))
    ncc_top_p10 = float(np.percentile(ncc_combined, 10))
    if ncc_top_p50 < NCC_TOP_P50_FLOOR:
        log(
            f"Calibración inestable: NCC top mediana = {ncc_top_p50:.3f} "
            f"(mínimo {NCC_TOP_P50_FLOOR})"
        )
        return None

    ncc_runner_combined = np.asarray(ncc_runner_a1 + ncc_runner_a2, dtype=np.float64)
    ncc_runner_up_p50 = (
        float(np.median(ncc_runner_combined)) if ncc_runner_combined.size else 0.0
    )

    obs_med_a1_x = float(np.median(obs_a1_x)) if obs_a1_x else float(anchor1[0])
    obs_med_a1_y = float(np.median(obs_a1_y)) if obs_a1_y else float(anchor1[1])
    obs_med_a2_x = float(np.median(obs_a2_x)) if obs_a2_x else float(anchor2[0])
    obs_med_a2_y = float(np.median(obs_a2_y)) if obs_a2_y else float(anchor2[1])

    # Click-off-perf diagnostic. The user click stays authoritative for the
    # warp target and for R4b's gate baseline (Decision C5); the warning
    # only flags that the click looks off-perf so the user can re-click if
    # they want a tighter lock.
    threshold = CLICK_OFF_PERF_FRACTION * perf_spacing
    for label, click, obs_x, obs_y in (
        ("anchor 1", anchor1, obs_med_a1_x, obs_med_a1_y),
        ("anchor 2", anchor2, obs_med_a2_x, obs_med_a2_y),
    ):
        dx = obs_x - float(click[0])
        dy = obs_y - float(click[1])
        dist = float(np.hypot(dx, dy))
        if dist > threshold:
            log(
                f"Calibración: el click en {label} parece off-perf "
                f"(separación click↔centroide observado = {dist:.1f}px, "
                f"umbral {threshold:.1f}px). Considera re-click."
            )

    log(
        f"Calibración OK: spacing={perf_spacing:.1f}px, "
        f"NCC top p50={ncc_top_p50:.3f}, effective_n={effective_n}"
    )

    return CalibrationState(
        perf_spacing=perf_spacing,
        template_a1=template_a1,
        template_a2=template_a2,
        anchor1_ref=(float(anchor1[0]), float(anchor1[1])),
        anchor2_ref=(float(anchor2[0]), float(anchor2[1])),
        ncc_top_p50=ncc_top_p50,
        ncc_top_p10=ncc_top_p10,
        ncc_runner_up_p50=ncc_runner_up_p50,
        effective_n=effective_n,
        sampled_indices=indices,
        observed_median_a1_x=obs_med_a1_x,
        observed_median_a1_y=obs_med_a1_y,
        observed_median_a2_x=obs_med_a2_x,
        observed_median_a2_y=obs_med_a2_y,
    )
