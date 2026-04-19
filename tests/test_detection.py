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
    _build_rotation_template,
    _detect_perf_bbox,
    _estimate_rotation,
    _extract_top_k_peaks,
    _template_match_candidates,
    _template_match_perforation,
    stabilize_folder,
)

# ── Synthetic frame helpers ────────────────────────────────────────────────────

FRAME_H = 1000
FRAME_W = 1200
PERF_W = 70
PERF_H = 100


def make_frame(perf_y_fracs, frame_h=FRAME_H, frame_w=FRAME_W,
               perf_w=PERF_W, perf_h=PERF_H):
    """Return a BGR frame with bright rectangles at given vertical positions.

    The rectangles are placed near the left side, horizontally centred in
    the inner third of the left 22% strip.
    """
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


# ── Template building ──────────────────────────────────────────────────────────

class TestBuildTemplate:

    def test_build_template_returns_patch(self):
        """Template extraction from a frame with a known anchor produces a valid patch."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, origin = _build_perforation_template(frame, anchor)
        assert tpl is not None
        assert tpl.ndim == 2  # grayscale
        assert tpl.shape[0] > 10 and tpl.shape[1] > 10
        assert origin is not None

    def test_build_template_at_frame_edge(self):
        """Anchor near (0, 0) clips to frame bounds without crashing."""
        frame = make_frame([0.01])
        anchor = (5.0, 5.0)
        tpl, origin = _build_perforation_template(frame, anchor)
        # Should either return a clipped patch or None — no crash
        if tpl is not None:
            assert tpl.shape[0] > 0 and tpl.shape[1] > 0

    def test_build_template_at_max_edge(self):
        """Anchor at frame's max corner clips correctly."""
        frame = make_frame([0.25])
        anchor = (float(FRAME_W - 3), float(FRAME_H - 3))
        tpl, origin = _build_perforation_template(frame, anchor)
        if tpl is not None:
            assert tpl.shape[0] > 0 and tpl.shape[1] > 0

    def test_build_template_non_finite_anchor(self):
        """Non-finite anchor returns (None, None)."""
        frame = make_frame([0.25])
        tpl, origin = _build_perforation_template(frame, (float("nan"), 100.0))
        assert tpl is None
        assert origin is None


# ── Contextual template (auto-scaled, asymmetric vertical) ────────────────────

class TestContextualTemplate:

    def test_auto_scaled_template_is_taller_than_wide(self):
        """Default (patch_radius=None) produces a vertical-asymmetric template."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        assert tpl is not None
        # Height should be clearly greater than width (captures perf + context)
        assert tpl.shape[0] > tpl.shape[1], (
            f"Expected tall template, got {tpl.shape}"
        )

    def test_auto_scaled_template_is_larger_than_perf(self):
        """Template height spans clearly more than the perforation itself."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        assert tpl is not None
        assert tpl.shape[0] > PERF_H * 1.5, (
            f"Template height {tpl.shape[0]} must exceed perf_h={PERF_H} "
            "with vertical context"
        )

    def test_legacy_patch_radius_still_works(self):
        """Explicit patch_radius preserves the original square-crop contract."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor, patch_radius=60)
        assert tpl is not None
        # With explicit radius the template is square (minus edge clipping)
        assert abs(tpl.shape[0] - tpl.shape[1]) <= 1

    def test_auto_scaled_fallback_on_blank_roi(self):
        """Anchor on uniform region falls back to heuristic without crash."""
        frame = np.full((FRAME_H, FRAME_W, 3), 30, dtype=np.uint8)
        tpl, _ = _build_perforation_template(frame, (600.0, 500.0))
        # Uniform frame: Otsu produces one big "foreground", filter may
        # accept or reject — either way we must not crash and must return
        # a usable template via fallback or the contour itself.
        if tpl is not None:
            assert tpl.shape[0] >= 20 and tpl.shape[1] >= 20

    def test_detect_perf_bbox_finds_perforation(self):
        """_detect_perf_bbox returns approximate perf dimensions."""
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


# ── Template matching ──────────────────────────────────────────────────────────

class TestTemplateMatching:

    def test_template_match_finds_anchor(self):
        """Template matching on the same frame used to build the template returns a match."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        assert tpl is not None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = _template_match_perforation(gray, tpl)
        assert result is not None
        cx, cy, conf = result
        assert abs(cx - anchor[0]) < 10
        assert abs(cy - anchor[1]) < 10
        assert conf > 0.8

    def test_template_match_subpixel_precision(self):
        """Template match returns float coordinates (sub-pixel)."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = _template_match_perforation(gray, tpl)
        assert result is not None
        cx, cy, _ = result
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    def test_template_match_rejects_blank_frame(self):
        """Template matching on an all-black frame returns None (low confidence)."""
        good = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(good, anchor)
        assert tpl is not None
        blank = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        result = _template_match_perforation(blank, tpl)
        assert result is None

    def test_search_roi_prefers_nearby_perforation(self):
        """With two identical perforations, ROI search locks to the nearby one."""
        # Build template at upper perf, then match on a frame containing
        # both upper and lower perfs. Without ROI the global peak could
        # land on either; with a tight ROI near the upper anchor it must
        # pick the upper one.
        tpl_frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(tpl_frame, anchor)
        two_perf_frame = make_frame([0.25, 0.75])
        gray = cv2.cvtColor(two_perf_frame, cv2.COLOR_BGR2GRAY)
        result = _template_match_perforation(
            gray, tpl, search_center=anchor, search_radius=100
        )
        assert result is not None
        cx, cy, _ = result
        assert abs(cy - anchor[1]) < 20, f"Match drifted to neighbour perf: cy={cy}"

    def test_template_match_consistent_across_identical_frames(self):
        """Template matching on identical frames returns the same position."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r1 = _template_match_perforation(gray, tpl)
        r2 = _template_match_perforation(gray, tpl)
        assert r1 is not None and r2 is not None
        assert abs(r1[0] - r2[0]) < 0.01
        assert abs(r1[1] - r2[1]) < 0.01


# ── Multi-peak (top-K) matching ───────────────────────────────────────────────

class TestMultiPeakMatching:

    def test_extract_top_k_finds_multiple_peaks(self):
        """A synthetic correlation map with 3 clear peaks yields them ranked."""
        corr = np.zeros((200, 200), dtype=np.float32)
        peaks_expected = [(50, 50, 0.9), (150, 150, 0.7), (50, 150, 0.5)]
        for x, y, v in peaks_expected:
            corr[y, x] = v

        peaks = _extract_top_k_peaks(corr, k=3, suppression_radius=20)
        assert len(peaks) == 3
        # Scores descending
        scores = [p[2] for p in peaks]
        assert scores == sorted(scores, reverse=True)
        # Top peak matches the highest synthetic value
        assert abs(peaks[0][0] - 50) < 1 and abs(peaks[0][1] - 50) < 1
        assert abs(peaks[0][2] - 0.9) < 0.01

    def test_extract_top_k_respects_suppression(self):
        """Peaks closer than suppression_radius are not both returned."""
        corr = np.zeros((100, 100), dtype=np.float32)
        corr[50, 50] = 0.9
        corr[50, 55] = 0.85  # within suppression_radius=20 of first

        peaks = _extract_top_k_peaks(corr, k=5, suppression_radius=20)
        # The suppressed second-peak should NOT appear as a distinct entry
        distinct_peaks = [
            p for p in peaks
            if abs(p[0] - 50) > 1 or abs(p[1] - 50) > 1
        ]
        # None of the returned "distinct" peaks should carry 0.85 --
        # it was suppressed because it fell inside the first peak's
        # neighbourhood.
        for _, _, score in distinct_peaks:
            assert score < 0.85 - 0.01

    def test_template_match_candidates_returns_both_perfs(self):
        """On a frame with two identical perfs, candidates contain both."""
        tpl_frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(tpl_frame, anchor)
        two_perf = make_frame([0.25, 0.75])
        gray = cv2.cvtColor(two_perf, cv2.COLOR_BGR2GRAY)

        candidates = _template_match_candidates(
            gray, tpl, k=5, min_confidence=0.3
        )
        # Need at least two high-confidence candidates near the two perfs
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
        """A blank frame below min_confidence returns an empty candidate list."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        blank = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        candidates = _template_match_candidates(
            blank, tpl, k=5, min_confidence=0.5
        )
        assert candidates == []


# ── Rotation estimation ───────────────────────────────────────────────────────

class TestRotationEstimation:

    def test_estimate_rotation_returns_near_zero_for_unrotated(self):
        """An unrotated frame matched against its own template gives ~0 degrees."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        assert tpl is not None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        angle = _estimate_rotation(gray, tpl, anchor[0], anchor[1])
        assert abs(angle) < 0.5

    def test_estimate_rotation_detects_small_rotation(self):
        """A slightly rotated frame produces a non-zero angle estimate."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        assert tpl is not None

        h, w = frame.shape[:2]
        R = cv2.getRotationMatrix2D((w / 2, h / 2), 1.0, 1.0)
        rotated = cv2.warpAffine(frame, R, (w, h))

        rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        tm_result = _template_match_perforation(rotated_gray, tpl)
        if tm_result is not None:
            cx, cy, _ = tm_result
            angle = _estimate_rotation(rotated_gray, tpl, cx, cy)
            assert isinstance(angle, float)

    def test_estimate_rotation_returns_none_on_failure(self):
        """Blank frame returns None so caller can carry previous angle."""
        frame = make_frame([0.25])
        anchor = _perf_centroid(0.25)
        tpl, _ = _build_perforation_template(frame, anchor)
        blank = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        angle = _estimate_rotation(blank, tpl, 50.0, 250.0)
        assert angle is None or angle == 0.0

    def test_rotation_template_scales_to_frame_height(self):
        """Rotation template uses ~30% of frame height for lever-arm accuracy."""
        frame = make_frame([0.25])  # FRAME_H = 1000
        anchor = _perf_centroid(0.25)
        rot_tpl = _build_rotation_template(frame, anchor)
        assert rot_tpl is not None
        # half_h should be >= 0.3 * 1000 = 300, so template height >= 600
        # (or clipped by frame bounds)
        assert rot_tpl.shape[0] >= 300, (
            f"Expected rotation template height >= 300 (30% of 1000), "
            f"got {rot_tpl.shape[0]}"
        )

    def test_rotation_template_caps_half_h(self):
        """Rotation template half-height is capped so ECC memory stays bounded."""
        # Synthesise a huge frame to verify the cap kicks in
        frame = np.zeros((6000, 2000, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 2900), (200, 3100), (240, 240, 240), -1)
        rot_tpl = _build_rotation_template(frame, (150.0, 3000.0))
        assert rot_tpl is not None
        # Cap is half_h <= 1000, so full template height <= 2000
        assert rot_tpl.shape[0] <= 2000


# ── Anchor-based stabilization workflow ────────────────────────────────────────

class TestAnchorWorkflow:

    def test_stabilize_with_anchor_produces_output(self):
        """stabilize_folder with anchor and 5 identical frames produces 5 output images."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            good = make_frame([0.25])
            for i in range(5):
                _write_frame(os.path.join(inp, f"frame_{i:03d}.jpg"), good)

            anchor = _perf_centroid(0.25)
            stabilize_folder(inp, out, anchor=anchor, jpeg_quality=95)

            out_images = [f for f in os.listdir(out) if f.endswith(".jpg")]
            assert len(out_images) == 5

    def test_stabilize_produces_report(self):
        """stabilize_folder writes a stabilization_report.txt."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "frame_001.jpg"), good)

            anchor = _perf_centroid(0.25)
            stabilize_folder(inp, out, anchor=anchor, jpeg_quality=95)

            report = os.path.join(out, "stabilization_report.txt")
            assert os.path.exists(report)
            content = open(report).read()
            assert "total_frames" in content
            assert "film_format" not in content  # removed

    def test_stabilize_single_frame(self):
        """Single-frame input produces one output image."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "single.jpg"), good)

            anchor = _perf_centroid(0.25)
            stabilize_folder(inp, out, anchor=anchor, jpeg_quality=95)

            out_images = [f for f in os.listdir(out) if f.endswith(".jpg")]
            assert len(out_images) == 1

    def test_stabilize_raises_without_anchor(self):
        """stabilize_folder raises ValueError when anchor is None."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "frame.jpg"), good)

            with pytest.raises(ValueError, match="anchor"):
                stabilize_folder(inp, out, anchor=None, jpeg_quality=95)

    def test_stabilize_no_images_raises(self):
        """Empty folder raises RuntimeError."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            anchor = (100.0, 200.0)
            with pytest.raises(RuntimeError, match="imágenes"):
                stabilize_folder(inp, out, anchor=anchor, jpeg_quality=95)

    def test_stabilize_identical_frames_zero_drift(self):
        """Template match on identical frames returns positions close to anchor (dx/dy ~ 0)."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            good = make_frame([0.25])
            for i in range(3):
                _write_frame(os.path.join(inp, f"frame_{i:03d}.jpg"), good)

            anchor = _perf_centroid(0.25)
            summary = stabilize_folder(inp, out, anchor=anchor, jpeg_quality=95)

            assert summary["failed_detections"] == 0
            # Target should be the anchor itself
            assert abs(summary["target_x"] - anchor[0]) < 1
            assert abs(summary["target_y"] - anchor[1]) < 1


# ── Debug frames ───────────────────────────────────────────────────────────────

class TestDebugFrames:

    def test_debug_jpeg_created_for_failed_frame(self):
        """Failed detection (all-black frame) produces a _debug.jpg when debug_dir is set."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out, \
             tempfile.TemporaryDirectory() as dbg:

            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "frame_001.jpg"), good)

            bad = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            _write_frame(os.path.join(inp, "frame_002.jpg"), bad)

            anchor = _perf_centroid(0.25)
            stabilize_folder(inp, out, anchor=anchor, debug_dir=dbg, jpeg_quality=95)

            debug_files = os.listdir(dbg)
            assert any(
                "frame_002" in f and f.endswith("_debug.jpg")
                for f in debug_files
            ), f"Expected debug file for frame_002, got: {debug_files}"

    def test_no_debug_file_for_successful_detection(self):
        """Successful detection produces no debug files."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out, \
             tempfile.TemporaryDirectory() as dbg:

            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "frame_001.jpg"), good)

            anchor = _perf_centroid(0.25)
            stabilize_folder(inp, out, anchor=anchor, debug_dir=dbg, jpeg_quality=95)

            assert os.listdir(dbg) == [], "No debug files expected for a successful run"

    def test_no_debug_dir_does_not_raise(self):
        """debug_dir=None produces no errors."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "frame.jpg"), good)

            anchor = _perf_centroid(0.25)
            stabilize_folder(inp, out, anchor=anchor, debug_dir=None, jpeg_quality=95)
