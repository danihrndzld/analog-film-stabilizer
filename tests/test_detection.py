"""Tests for the anchor-based stabilization pipeline.

These tests use synthetic frames built with NumPy/OpenCV — no sample images
needed.  The user provides an anchor point (x, y) on the first frame, and
template matching tracks that point across all frames.
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from perforation_stabilizer_app import (
    _build_perforation_template,
    _detect_perf_bbox,
    _detect_perf_spacing,
    _extract_top_k_peaks,
    _FrameOutcome,
    _locate_anchor_in_frame,
    _MotionPredictor,
    _rank_candidates,
    _template_match_candidates,
    stabilize_folder,
)

# ── Synthetic frame helpers ────────────────────────────────────────────────────

FRAME_H = 1000
FRAME_W = 1200
PERF_W = 70
PERF_H = 100

# Three perforation positions: perf_spacing must be >= 1.5 * PERF_H for
# _detect_perf_spacing to accept the estimate. With FRAME_H=1000 and
# PERF_H=100 these fractions give spacing = 300 px (>> 150).
TRI_PERF = [0.2, 0.5, 0.8]


def make_frame(perf_y_fracs, frame_h=FRAME_H, frame_w=FRAME_W,
               perf_w=PERF_W, perf_h=PERF_H):
    """Return a BGR frame with bright rectangles at given vertical positions."""
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    rw = max(50, int(frame_w * 0.22))
    cx = rw // 3

    for yf in perf_y_fracs:
        cy = int(frame_h * yf)
        x1, y1 = cx - perf_w // 2, cy - perf_h // 2
        x2, y2 = cx + perf_w // 2, cy + perf_h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), -1)

    return frame


def _write_frame(path, frame):
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _perf_centroid(y_frac=0.25, frame_h=FRAME_H, frame_w=FRAME_W):
    """Return the approximate centroid (x, y) of a perforation rectangle."""
    rw = max(50, int(frame_w * 0.22))
    cx = rw // 3
    cy = int(frame_h * y_frac)
    return (float(cx), float(cy))


def _tri_frame():
    """Three-perforation frame suitable for full stabilize_folder runs."""
    return make_frame(TRI_PERF)


def _tri_anchor():
    """Middle-perforation anchor for _tri_frame()."""
    return _perf_centroid(0.5)


# ── Template building ──────────────────────────────────────────────────────────

class TestBuildTemplate:

    def test_build_template_returns_patch(self):
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, anchor_in_tpl = _build_perforation_template(frame, anchor)
        assert tpl is not None
        assert tpl.ndim == 2  # grayscale
        assert tpl.shape[0] > 10 and tpl.shape[1] > 10
        assert anchor_in_tpl is not None
        assert len(anchor_in_tpl) == 2

    def test_build_template_anchor_offset_matches_input(self):
        """anchor_in_tpl places the anchor inside the template at (cx-x0, cy-y0)."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, anchor_in_tpl = _build_perforation_template(frame, anchor)
        assert tpl is not None
        ax, ay = anchor_in_tpl
        # Must fall inside the template
        assert 0 <= ax <= tpl.shape[1]
        assert 0 <= ay <= tpl.shape[0]

    def test_build_template_at_frame_edge(self):
        """Anchor near (0, 0) clips; anchor_in_tpl reflects the clipped offset."""
        frame = make_frame([0.01])
        anchor = (5.0, 5.0)
        tpl, anchor_in_tpl = _build_perforation_template(frame, anchor)
        if tpl is not None:
            assert tpl.shape[0] > 0 and tpl.shape[1] > 0
            ax, ay = anchor_in_tpl
            # Clipped to frame origin: anchor offset inside tpl equals anchor coord
            assert ax == pytest.approx(5.0)
            assert ay == pytest.approx(5.0)

    def test_build_template_at_max_edge(self):
        frame = make_frame([0.25])
        anchor = (float(FRAME_W - 3), float(FRAME_H - 3))
        tpl, anchor_in_tpl = _build_perforation_template(frame, anchor)
        if tpl is not None:
            assert tpl.shape[0] > 0 and tpl.shape[1] > 0
            ax, ay = anchor_in_tpl
            # Anchor should land near the right/bottom of the template
            assert ax > tpl.shape[1] // 2
            assert ay > tpl.shape[0] // 2

    def test_build_template_non_finite_anchor(self):
        frame = make_frame([0.25])
        tpl, anchor_in_tpl = _build_perforation_template(frame, (float("nan"), 100.0))
        assert tpl is None
        assert anchor_in_tpl is None


# ── Contextual template (auto-scaled, asymmetric vertical) ────────────────────

class TestContextualTemplate:

    def test_auto_scaled_template_is_taller_than_wide(self):
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        assert tpl is not None
        assert tpl.shape[0] > tpl.shape[1]

    def test_auto_scaled_template_is_larger_than_perf(self):
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        assert tpl is not None
        assert tpl.shape[0] > PERF_H * 1.5

    def test_legacy_patch_radius_still_works(self):
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor, patch_radius=60)
        assert tpl is not None
        assert abs(tpl.shape[0] - tpl.shape[1]) <= 1

    def test_auto_scaled_fallback_on_blank_roi(self):
        frame = np.full((FRAME_H, FRAME_W, 3), 30, dtype=np.uint8)
        tpl, _ = _build_perforation_template(frame, (600.0, 500.0))
        if tpl is not None:
            assert tpl.shape[0] >= 20 and tpl.shape[1] >= 20

    def test_detect_perf_bbox_finds_perforation(self):
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        bbox = _detect_perf_bbox(frame, anchor)
        assert bbox is not None
        bw, bh = bbox
        assert abs(bw - PERF_W) < 10
        assert abs(bh - PERF_H) < 10

    def test_detect_perf_bbox_returns_none_on_non_finite_anchor(self):
        frame = make_frame([0.25])
        bbox = _detect_perf_bbox(frame, (float("nan"), 100.0))
        assert bbox is None


# ── Template matching (candidates with k=1) ───────────────────────────────────

class TestTemplateMatching:

    def test_candidates_finds_anchor(self):
        """Top candidate on the same frame used to build the template matches the anchor."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, anchor_in_tpl = _build_perforation_template(frame, anchor)
        assert tpl is not None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cands = _template_match_candidates(
            gray, tpl, k=1, anchor_in_tpl=anchor_in_tpl
        )
        assert cands, "Expected at least one candidate"
        cx, cy, conf = cands[0]
        assert abs(cx - anchor[0]) < 10
        assert abs(cy - anchor[1]) < 10
        assert conf > 0.8

    def test_candidates_subpixel_precision(self):
        """Candidates return float coordinates (sub-pixel)."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, anchor_in_tpl = _build_perforation_template(frame, anchor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cands = _template_match_candidates(
            gray, tpl, k=1, anchor_in_tpl=anchor_in_tpl
        )
        assert cands
        cx, cy, _ = cands[0]
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    def test_candidates_rejects_blank_frame(self):
        """Candidates on an all-black frame return [] under a min_confidence floor."""
        good = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, anchor_in_tpl = _build_perforation_template(good, anchor)
        assert tpl is not None
        blank = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        cands = _template_match_candidates(
            blank, tpl, k=1, min_confidence=0.5, anchor_in_tpl=anchor_in_tpl
        )
        assert cands == []

    def test_search_roi_prefers_nearby_perforation(self):
        """With two identical perforations, ROI search locks to the nearby one."""
        tpl_frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, anchor_in_tpl = _build_perforation_template(tpl_frame, anchor)
        two_perf_frame = make_frame([0.25, 0.75])
        gray = cv2.cvtColor(two_perf_frame, cv2.COLOR_BGR2GRAY)
        cands = _template_match_candidates(
            gray, tpl, k=1,
            search_center=anchor, search_radius=100,
            anchor_in_tpl=anchor_in_tpl,
        )
        assert cands, "Expected a candidate within the ROI"
        cx, cy, _ = cands[0]
        assert abs(cy - anchor[1]) < 20, f"Match drifted to neighbour perf: cy={cy}"

    def test_candidates_consistent_across_identical_frames(self):
        """Repeated matches on the same frame return the same position."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, anchor_in_tpl = _build_perforation_template(frame, anchor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        c1 = _template_match_candidates(gray, tpl, k=1, anchor_in_tpl=anchor_in_tpl)
        c2 = _template_match_candidates(gray, tpl, k=1, anchor_in_tpl=anchor_in_tpl)
        assert c1 and c2
        assert abs(c1[0][0] - c2[0][0]) < 0.01
        assert abs(c1[0][1] - c2[0][1]) < 0.01


# ── Multi-peak (top-K) matching ───────────────────────────────────────────────

class TestMultiPeakMatching:

    def test_extract_top_k_finds_multiple_peaks(self):
        corr = np.zeros((200, 200), dtype=np.float32)
        peaks_expected = [(50, 50, 0.9), (150, 150, 0.7), (50, 150, 0.5)]
        for x, y, v in peaks_expected:
            corr[y, x] = v

        peaks = _extract_top_k_peaks(corr, k=3, suppression_radius=20)
        assert len(peaks) == 3
        scores = [p[2] for p in peaks]
        assert scores == sorted(scores, reverse=True)
        assert abs(peaks[0][0] - 50) < 1 and abs(peaks[0][1] - 50) < 1
        assert abs(peaks[0][2] - 0.9) < 0.01

    def test_extract_top_k_respects_suppression(self):
        corr = np.zeros((100, 100), dtype=np.float32)
        corr[50, 50] = 0.9
        corr[50, 55] = 0.85  # within suppression_radius=20 of first

        peaks = _extract_top_k_peaks(corr, k=5, suppression_radius=20)
        distinct_peaks = [
            p for p in peaks
            if abs(p[0] - 50) > 1 or abs(p[1] - 50) > 1
        ]
        for _, _, score in distinct_peaks:
            assert score < 0.85 - 0.01

    def test_extract_top_k_consumes_corr_map(self):
        """In-place suppression: the input surface is mutated (hot-path no-copy)."""
        corr = np.zeros((100, 100), dtype=np.float32)
        corr[50, 50] = 0.9
        before_max = float(corr.max())
        _extract_top_k_peaks(corr, k=1, suppression_radius=20)
        # Peak should have been suppressed in place
        assert float(corr.max()) < before_max

    def test_extract_top_k_subpixel_survives_suppression_neighbours(self):
        """Sub-pixel guard: a peak adjacent to a prior suppression window
        must still return a finite integer-or-refined location, not NaN."""
        corr = np.zeros((100, 100), dtype=np.float32)
        corr[20, 20] = 0.9  # first peak
        corr[20, 42] = 0.7  # second peak; its x=41 neighbour will be poisoned
                             # by the first peak's suppression window (radius=20)
        peaks = _extract_top_k_peaks(corr, k=2, suppression_radius=20)
        assert len(peaks) >= 2
        for sx, sy, _ in peaks:
            assert np.isfinite(sx) and np.isfinite(sy)

    def test_template_match_candidates_returns_both_perfs(self):
        tpl_frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, anchor_in_tpl = _build_perforation_template(tpl_frame, anchor)
        two_perf = make_frame([0.25, 0.75])
        gray = cv2.cvtColor(two_perf, cv2.COLOR_BGR2GRAY)

        candidates = _template_match_candidates(
            gray, tpl, k=5, min_confidence=0.3, anchor_in_tpl=anchor_in_tpl,
        )
        upper = _perf_centroid(0.25)
        lower = _perf_centroid(0.75)
        matched_upper = any(
            abs(c[0] - upper[0]) < 15 and abs(c[1] - upper[1]) < 15
            for c in candidates
        )
        matched_lower = any(
            abs(c[0] - lower[0]) < 15 and abs(c[1] - lower[1]) < 15
            for c in candidates
        )
        assert matched_upper and matched_lower, (
            f"Expected both perfs in candidates, got: {candidates}"
        )

    def test_template_match_candidates_empty_on_blank_frame(self):
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, anchor_in_tpl = _build_perforation_template(frame, anchor)
        blank = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        candidates = _template_match_candidates(
            blank, tpl, k=5, min_confidence=0.5, anchor_in_tpl=anchor_in_tpl,
        )
        assert candidates == []


# ── Ranking and ambiguity ─────────────────────────────────────────────────────

class TestRankingAndAmbiguity:

    def test_motion_predictor_smooths_delta(self):
        p = _MotionPredictor(initial_pos=(100.0, 200.0), alpha=0.5)
        p.update((110.0, 200.0))
        p.update((120.0, 200.0))
        px, py = p.predict()
        assert 125.0 <= px <= 135.0
        assert abs(py - 200.0) < 1e-6

    def test_motion_predictor_alpha_one_tracks_last_delta_only(self):
        """alpha=1.0 disables EMA: delta is always last frame's delta."""
        p = _MotionPredictor(initial_pos=(0.0, 0.0), alpha=1.0)
        p.update((10.0, 0.0))  # delta = 10
        p.update((10.0, 0.0))  # no motion → delta becomes 0
        px, _ = p.predict()
        assert abs(px - 10.0) < 1e-6

    def test_motion_predictor_alpha_zero_never_updates_delta(self):
        """alpha=0.0 freezes delta at its initial (0, 0) value."""
        p = _MotionPredictor(initial_pos=(0.0, 0.0), alpha=0.0)
        p.update((10.0, 0.0))
        p.update((20.0, 0.0))
        assert p.delta == (0.0, 0.0)
        px, _ = p.predict()
        # last_pos updates but delta stays zero, so predict ≡ last_pos
        assert abs(px - 20.0) < 1e-6

    def test_motion_predictor_bootstrap_before_first_update(self):
        """Before any update, predict() returns the initial_pos exactly."""
        p = _MotionPredictor(initial_pos=(42.0, 17.0), alpha=0.5)
        px, py = p.predict()
        assert (px, py) == (42.0, 17.0)

    def test_predictor_update_preserves_ema(self):
        p = _MotionPredictor(initial_pos=(0.0, 0.0), alpha=0.5)
        p.update((10.0, 0.0))
        first_delta = p.delta
        p.update((10.0, 0.0))
        assert abs(p.delta[0]) <= abs(first_delta[0]) + 1e-6

    def test_closer_candidate_wins_with_similar_ncc(self):
        candidates = [(100, 100, 0.90), (100, 300, 0.88)]
        predicted = (100, 110)
        ranked, ambiguous = _rank_candidates(candidates, predicted, perf_spacing=200)
        assert ranked[0][0] == 100 and ranked[0][1] == 100
        assert not ambiguous

    def test_higher_ncc_wins_equidistant(self):
        candidates = [(100, 100, 0.95), (100, 300, 0.60)]
        predicted = (100, 200)
        ranked, _ = _rank_candidates(candidates, predicted, perf_spacing=200)
        assert ranked[0][2] == 0.95

    def test_ambiguity_flagged_on_twin_peaks(self):
        candidates = [(100, 100, 0.90), (100, 300, 0.90)]
        predicted = (100, 200)
        _, ambiguous = _rank_candidates(candidates, predicted, perf_spacing=200)
        assert ambiguous

    def test_single_candidate_never_ambiguous(self):
        candidates = [(100, 100, 0.90)]
        _, ambiguous = _rank_candidates(candidates, (100, 100), perf_spacing=200)
        assert not ambiguous

    def test_empty_candidates_returns_empty(self):
        ranked, ambiguous = _rank_candidates([], (0, 0), perf_spacing=200)
        assert ranked == []
        assert not ambiguous

    def test_rank_candidates_with_none_perf_spacing(self):
        """perf_spacing=None falls back to NCC-only ranking; never flags ambiguity."""
        candidates = [(100, 100, 0.88), (100, 300, 0.90)]
        ranked, ambiguous = _rank_candidates(candidates, (0, 0), perf_spacing=None)
        assert len(ranked) == 2
        assert ranked[0][2] == 0.90
        assert not ambiguous

    def test_rank_candidates_duplicate_positions_are_index_distinct(self):
        """Two candidates sharing coords (e.g. rounding edge cases) don't collapse
        into one via float-equality self-skip in the dominance loop."""
        candidates = [(100.0, 100.0, 0.90), (100.0, 100.0, 0.80)]
        ranked, _ = _rank_candidates(candidates, (100.0, 100.0), perf_spacing=200)
        # Both entries preserved (index-based self-skip)
        assert len(ranked) == 2

    def test_detect_perf_spacing_finds_three_perfs(self):
        frame = make_frame(TRI_PERF)  # spacing ~ 300 px
        anchor = _perf_centroid(0.5)
        spacing = _detect_perf_spacing(frame, anchor)
        assert spacing is not None
        assert 250 <= spacing <= 350

    def test_detect_perf_spacing_returns_none_when_only_one_perf(self):
        frame = make_frame([0.5])
        anchor = _perf_centroid(0.5)
        spacing = _detect_perf_spacing(frame, anchor)
        assert spacing is None

    def test_detect_perf_spacing_returns_none_when_only_two_perfs(self):
        """Two perfs give one delta — not enough signal to call it periodic."""
        frame = make_frame([0.3, 0.7])
        anchor = _perf_centroid(0.3)
        spacing = _detect_perf_spacing(frame, anchor)
        assert spacing is None

    def test_detect_perf_spacing_rejects_spacing_below_1_5_perf_h(self):
        """Median delta below 1.5 * perf_h is rejected (contour fragments, not perfs)."""
        # PERF_H = 100 → 1.5 * perf_h = 150. Place three perfs 120 px apart
        # in absolute pixel terms (fractions tuned for FRAME_H=1000).
        frame = make_frame([0.3, 0.42, 0.54])  # deltas ~120 px < 150
        anchor = _perf_centroid(0.3)
        spacing = _detect_perf_spacing(frame, anchor)
        assert spacing is None


# ── _locate_anchor_in_frame integration ──────────────────────────────────────

class TestLocateAnchorInFrame:
    def _setup(self):
        frame = _tri_frame()
        anchor = _tri_anchor()
        tpl, anchor_in_tpl = _build_perforation_template(frame, anchor)
        predictor = _MotionPredictor(initial_pos=anchor, alpha=0.6)
        perf_spacing = _detect_perf_spacing(frame, anchor)
        assert perf_spacing is not None
        search_radius = int(min(max(perf_spacing * 0.45, 80), perf_spacing - 40))
        return frame, tpl, anchor_in_tpl, predictor, search_radius, perf_spacing

    def test_success_branch_returns_pt_near_anchor(self):
        frame, tpl, a_in_tpl, pred, sr, ps = self._setup()
        outcome = _locate_anchor_in_frame(frame, tpl, a_in_tpl, pred, sr, ps)
        assert isinstance(outcome, _FrameOutcome)
        assert outcome.pt is not None
        assert not outcome.ambiguous
        assert not outcome.motion_rejected
        cx, cy = outcome.pt
        anchor = _tri_anchor()
        assert abs(cx - anchor[0]) < 10
        assert abs(cy - anchor[1]) < 10

    def test_motion_rejected_branch_on_blank_frame(self):
        _, tpl, a_in_tpl, pred, sr, ps = self._setup()
        blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        outcome = _locate_anchor_in_frame(blank, tpl, a_in_tpl, pred, sr, ps)
        assert outcome.pt is None
        assert outcome.motion_rejected
        assert not outcome.ambiguous

    def test_motion_gate_rejects_candidate_far_from_prediction(self):
        """Unit 6 gate: a lone top candidate further than 0.5×perf_spacing
        from the predictor must be rejected as motion_rejected so the
        predictor never locks onto a neighbour-perf swap."""
        frame, tpl, a_in_tpl, _pred, _sr, ps = self._setup()
        # Place the predictor at the top of the frame, far above every
        # perforation. The nearest real perf (top one, at 0.2×FRAME_H) is
        # > 0.5×perf_spacing away. Widen search_radius so the ROI still
        # covers that perf; otherwise we'd hit the empty-ROI branch.
        top_perf_y = int(TRI_PERF[0] * FRAME_H)  # truth: top perforation
        assert top_perf_y > 0.5 * ps, "test setup invariant"
        off_predictor = _MotionPredictor(initial_pos=(_tri_anchor()[0], 0.0), alpha=0.6)
        wide_sr = top_perf_y + 40  # covers the perf; still > 0.5×ps from predictor
        outcome = _locate_anchor_in_frame(
            frame, tpl, a_in_tpl, off_predictor, wide_sr, ps
        )
        assert outcome.motion_rejected
        assert outcome.pt is None
        assert not outcome.ambiguous
        # Predictor state must not have been nudged by a rejected candidate.
        assert off_predictor.predict() == (_tri_anchor()[0], 0.0)


# ── Search radius invariant ───────────────────────────────────────────────────

class TestSearchRadiusInvariant:
    @pytest.mark.parametrize("perf_spacing", [200, 240, 300, 400, 800])
    def test_search_radius_strictly_less_than_spacing(self, perf_spacing):
        """The search radius must never reach the neighbour perforation."""
        # Mirror the production formula exactly:
        sr = int(min(max(perf_spacing * 0.45, 80), perf_spacing - 40))
        assert sr < perf_spacing
        # And it must leave headroom for jitter (>= 80 px for reasonable scans,
        # unless spacing is so tight that the upper cap forces it lower)
        assert sr >= min(80, perf_spacing - 40)


# ── Anchor-based stabilization workflow ────────────────────────────────────────

class TestAnchorWorkflow:

    def test_stabilize_with_anchor_produces_output(self):
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            good = _tri_frame()
            for i in range(5):
                _write_frame(os.path.join(inp, f"frame_{i:03d}.jpg"), good)

            stabilize_folder(inp, out, anchor=_tri_anchor(), jpeg_quality=95)

            out_images = [f for f in os.listdir(out) if f.endswith(".jpg")]
            assert len(out_images) == 5

    def test_stabilize_produces_report(self):
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            _write_frame(os.path.join(inp, "frame_001.jpg"), _tri_frame())

            stabilize_folder(inp, out, anchor=_tri_anchor(), jpeg_quality=95)

            report = os.path.join(out, "stabilization_report.txt")
            assert os.path.exists(report)
            content = open(report).read()
            assert "total_frames" in content
            assert "film_format" not in content

    def test_stabilize_single_frame(self):
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            _write_frame(os.path.join(inp, "single.jpg"), _tri_frame())

            stabilize_folder(inp, out, anchor=_tri_anchor(), jpeg_quality=95)

            out_images = [f for f in os.listdir(out) if f.endswith(".jpg")]
            assert len(out_images) == 1

    def test_stabilize_raises_without_anchor(self):
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            _write_frame(os.path.join(inp, "frame.jpg"), _tri_frame())

            with pytest.raises(ValueError, match="anchor"):
                stabilize_folder(inp, out, anchor=None, jpeg_quality=95)

    def test_stabilize_no_images_raises(self):
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            with pytest.raises(RuntimeError, match="imágenes"):
                stabilize_folder(inp, out, anchor=(100.0, 200.0), jpeg_quality=95)

    def test_stabilize_raises_when_perf_spacing_undetectable(self):
        """Single-perforation frames can't yield a spacing estimate → raise."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            _write_frame(os.path.join(inp, "frame.jpg"), make_frame([0.5]))
            with pytest.raises(RuntimeError, match="espaciado"):
                stabilize_folder(
                    inp, out, anchor=_perf_centroid(0.5), jpeg_quality=95
                )

    def test_stabilize_identical_frames_zero_drift(self):
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            good = _tri_frame()
            for i in range(3):
                _write_frame(os.path.join(inp, f"frame_{i:03d}.jpg"), good)

            anchor = _tri_anchor()
            summary = stabilize_folder(inp, out, anchor=anchor, jpeg_quality=95)

            assert summary["failed_detections"] == 0
            assert abs(summary["target_x"] - anchor[0]) < 1
            assert abs(summary["target_y"] - anchor[1]) < 1


# ── Debug frames ───────────────────────────────────────────────────────────────

class TestDebugFrames:

    def test_debug_jpeg_created_for_failed_frame(self):
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out, \
             tempfile.TemporaryDirectory() as dbg:

            _write_frame(os.path.join(inp, "frame_001.jpg"), _tri_frame())

            bad = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            _write_frame(os.path.join(inp, "frame_002.jpg"), bad)

            stabilize_folder(
                inp, out, anchor=_tri_anchor(), debug_dir=dbg, jpeg_quality=95
            )

            debug_files = os.listdir(dbg)
            assert any(
                "frame_002" in f and f.endswith("_debug.jpg")
                for f in debug_files
            ), f"Expected debug file for frame_002, got: {debug_files}"

    def test_no_debug_file_for_successful_detection(self):
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out, \
             tempfile.TemporaryDirectory() as dbg:

            _write_frame(os.path.join(inp, "frame_001.jpg"), _tri_frame())

            stabilize_folder(
                inp, out, anchor=_tri_anchor(), debug_dir=dbg, jpeg_quality=95
            )

            assert os.listdir(dbg) == [], "No debug files expected for a successful run"

    def test_no_debug_dir_does_not_raise(self):
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            _write_frame(os.path.join(inp, "frame.jpg"), _tri_frame())

            stabilize_folder(
                inp, out, anchor=_tri_anchor(), debug_dir=None, jpeg_quality=95
            )
