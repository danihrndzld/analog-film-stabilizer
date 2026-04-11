"""Tests for perforation detection algorithm (Units 1-3).

These tests use synthetic frames built with NumPy/OpenCV — no sample images
needed.  Frame dimensions are chosen so that perforation blobs satisfy all
default filter thresholds (area ≥ 5000, aspect 0.40-1.20, fill ≥ 0.75).

Anchor point convention: the detection algorithm returns the BOTTOM-RIGHT
corner of the perforation bounding rect (x+bw, y+bh), which is the inner
corner closest to the exposed image area. This provides more stable
anchoring than centroid because perforation size can vary along the film.
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from perforation_stabilizer_app import (
    _annotate_roi_preview,
    _best_contour,
    detect_perforation,
    stabilize_folder,
)

# ── Synthetic frame helpers ────────────────────────────────────────────────────

# Standard synthetic frame size used across most tests.
# roi_w = int(1200 * 0.22) = 264 px
FRAME_H = 1000
FRAME_W = 1200

# Perforation rectangle size: aspect=70/100=0.70, area=7000 — passes all
# default filters (area ≥ 5000, aspect 0.40-1.20, fill=1.0 ≥ 0.75).
PERF_W = 70
PERF_H = 100


def _roi_w(frame_w=FRAME_W, roi_ratio=0.22):
    return max(50, int(frame_w * roi_ratio))


def make_frame(perf_y_fracs, frame_h=FRAME_H, frame_w=FRAME_W,
               perf_w=PERF_W, perf_h=PERF_H):
    """Return a BGR frame with bright rectangles at given vertical positions.

    The rectangles are placed in the left ROI strip, horizontally centred in
    the inner third of that strip so their centroids are well within the 70 %
    roi_w positional filter.

    Args:
        perf_y_fracs: iterable of y positions as fractions of frame_h.
    """
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    rw = _roi_w(frame_w)
    cx = rw // 3  # keeps centroid well within the 70 % filter

    for yf in perf_y_fracs:
        cy = int(frame_h * yf)
        x1, y1 = cx - perf_w // 2, cy - perf_h // 2
        x2, y2 = cx + perf_w // 2, cy + perf_h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), -1)

    return frame


def make_binary_roi(h=300, w=500):
    """Return an all-black binary image."""
    return np.zeros((h, w), dtype=np.uint8)


# ── Unit 1: _best_contour ─────────────────────────────────────────────────────

class TestBestContour:

    def test_happy_path_solid_rect(self):
        """Solid rect satisfying all default filters → bottom-right corner returned."""
        img = make_binary_roi(h=300, w=500)
        # 100×100 at (50, 50) to (150, 150): area=10000, aspect=1.0, fill=1.0
        # Bottom-right corner is (150, 150), left edge x=50 < 350 (70% of 500)
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
        result = _best_contour(img, roi_w=500)
        assert result is not None
        cx, cy = result
        # Anchor is bottom-right corner (x+bw, y+bh) = (150, 150)
        assert abs(cx - 150) < 3
        assert abs(cy - 150) < 3

    def test_aspect_min_boundary_rejected_at_040_accepted_at_025(self):
        """Contour with aspect≈0.30 is rejected at default aspect_min=0.40
        but accepted when aspect_min=0.25 is passed."""
        img = make_binary_roi(h=500, w=500)
        # bw=60, bh=200 → aspect=0.30, area=12000, fill=1.0, cx=90 < 350
        cv2.rectangle(img, (60, 150), (120, 350), 255, -1)

        assert _best_contour(img, roi_w=500) is None  # default aspect_min=0.40
        result = _best_contour(img, roi_w=500, aspect_min=0.25)
        assert result is not None

    def test_fill_min_boundary_rejected_at_075_accepted_at_065(self):
        """L-shaped contour with fill≈0.72 is rejected at fill_min=0.75
        but accepted at fill_min=0.65."""
        img = make_binary_roi(h=500, w=500)
        # Full 200×200 rect at (30, 150), then erase top-right 106×106 corner.
        # Remaining L-shape: area≈40000-11236=28764, bb=200×200=40000,
        # fill≈0.72, aspect=1.0, cx≈80 (well within 70 % limit).
        cv2.rectangle(img, (30, 150), (230, 350), 255, -1)
        cv2.rectangle(img, (124, 150), (230, 256), 0, -1)

        assert _best_contour(img, roi_w=500, aspect_min=0.25) is None  # fill 0.72 < 0.75
        # Disable solidity filter (L-shape has low solidity by design) to
        # isolate the fill_min boundary test.
        result = _best_contour(img, roi_w=500, aspect_min=0.25, fill_min=0.65,
                               solidity_min=0.0)
        assert result is not None

    def test_x_position_filter_uses_left_edge(self):
        """Contour whose left edge exceeds 70% roi_w is rejected."""
        roi_w = 500  # 70 % threshold = 350
        img = make_binary_roi(h=500, w=roi_w)
        # x=360 (72 %), bw=100, bh=100 → left edge past threshold
        cv2.rectangle(img, (360, 150), (460, 250), 255, -1)
        # area=100*100=10000, aspect=1.0, fill=1.0 — x-position filter rejects.
        result = _best_contour(img, roi_w=roi_w)
        assert result is None

    def test_x_position_filter_accepts_contour_near_threshold(self):
        """Contour whose left edge is at 68% roi_w (under 70%) is accepted."""
        roi_w = 500  # 70 % threshold = 350
        img = make_binary_roi(h=500, w=roi_w)
        # x=340 (68 %), bw=100, bh=100 → left edge under threshold
        cv2.rectangle(img, (340, 150), (440, 250), 255, -1)
        # area=100*100=10000, aspect=1.0, fill=1.0 — accepted
        result = _best_contour(img, roi_w=roi_w)
        assert result is not None

    def test_rejection_reasons_collected(self):
        """collect_rejections=True returns a reason string for each failure."""
        img = make_binary_roi(h=300, w=500)
        cv2.rectangle(img, (10, 10), (20, 20), 255, -1)  # 10×10 = 100 < 5000

        result, rejections = _best_contour(img, roi_w=500, collect_rejections=True)
        assert result is None
        assert len(rejections) == 1
        x, y, bw, bh, reason = rejections[0]
        assert reason == "area"

    # Return-type matrix ---------------------------------------------------

    def test_top1_no_collect_returns_none_on_empty(self):
        assert _best_contour(make_binary_roi(), roi_w=500) is None

    def test_top2_no_collect_returns_list(self):
        result = _best_contour(make_binary_roi(), roi_w=500, top_n=2)
        assert isinstance(result, list)

    def test_top1_with_collect_returns_tuple(self):
        result = _best_contour(make_binary_roi(), roi_w=500, collect_rejections=True)
        assert isinstance(result, tuple) and len(result) == 2
        pt, rej = result
        assert pt is None
        assert isinstance(rej, list)

    def test_top2_with_collect_returns_tuple_of_lists(self):
        result = _best_contour(make_binary_roi(), roi_w=500, top_n=2,
                               collect_rejections=True)
        assert isinstance(result, tuple) and len(result) == 2
        candidates, rej = result
        assert isinstance(candidates, list)
        assert isinstance(rej, list)


# ── Unit 2: _annotate_roi_preview ────────────────────────────────────────────

class TestAnnotateRoiPreview:

    def _make_roi(self, h=600, w=200):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_detected_returns_same_shape_and_dtype(self):
        roi = self._make_roi()
        result = _annotate_roi_preview(roi, anchor=(80, 300), rejections=[], frame_name="test.jpg")
        assert result.shape == roi.shape
        assert result.dtype == roi.dtype

    def test_detected_has_green_pixels_near_anchor(self):
        """Green crosshair should produce non-zero green channel pixels at anchor coords."""
        roi = self._make_roi()
        cx, cy = 80, 300
        result = _annotate_roi_preview(roi, anchor=(cx, cy), rejections=[])
        # Green channel (index 1) along the horizontal crosshair line at cy
        assert result[cy, cx, 1] > 100  # green component prominent

    def test_no_detection_returns_same_shape(self):
        roi = self._make_roi()
        rejections = [(10, 50, 40, 60, "area")]
        result = _annotate_roi_preview(roi, anchor=None, rejections=rejections)
        assert result.shape == roi.shape

    def test_no_detection_has_red_pixels_from_rejection_box(self):
        """Red channel should be prominent on rejection bounding rect."""
        roi = self._make_roi()
        rejections = [(10, 50, 40, 60, "fill")]
        result = _annotate_roi_preview(roi, anchor=None, rejections=rejections)
        # Red channel (index 2) should be non-zero near the rejection box
        region = result[50:111, 10:51, 2]
        assert region.max() > 100

    def test_does_not_modify_input(self):
        """The function must not mutate the input roi_bgr array."""
        roi = self._make_roi()
        original = roi.copy()
        _annotate_roi_preview(roi, anchor=(50, 200), rejections=[])
        assert np.array_equal(roi, original)


# ── Unit 3: detect_perforation ────────────────────────────────────────────────

class TestDetectPerforation:

    def test_8mm_two_perfs_returns_topmost_not_midpoint(self):
        """Two well-separated perforations → anchor is the TOPMOST perf's bottom-right corner."""
        frame = make_frame([0.25, 0.75])
        pt = detect_perforation(frame, film_format="8mm")
        assert pt is not None
        # Topmost perf centroid y ≈ 0.25 × FRAME_H = 250
        # Bottom edge = centroid_y + PERF_H/2 = 250 + 50 = 300 (0.30 of FRAME_H)
        assert abs(pt[1] - FRAME_H * 0.30) < FRAME_H * 0.08
        # Explicitly verify it's NOT the midpoint
        assert abs(pt[1] - FRAME_H * 0.50) > FRAME_H * 0.10

    def test_8mm_anchor_consistent_single_vs_double_perf(self):
        """Frame with 2 perfs and frame with only top perf both return anchor near same y."""
        frame_two = make_frame([0.25, 0.75])
        frame_one = make_frame([0.25])  # only top perf visible
        pt_two = detect_perforation(frame_two, film_format="8mm")
        pt_one = detect_perforation(frame_one, film_format="8mm")
        assert pt_two is not None and pt_one is not None
        # Both anchors should be near the top perf's bottom edge — delta < 20 px
        # (single-perf fallback and two-perf detection may use slightly different code paths)
        assert abs(pt_two[1] - pt_one[1]) < 20

    def test_8mm_perf_at_boundary_of_zone_detected(self):
        """Perf whose bottom edge is just under 35% zone threshold is detected."""
        # Bottom edge at 34% of frame height (340px) → cy < 350 threshold
        # Perf centroid at 29% (290px), bottom edge at 290 + 50 = 340
        frame = make_frame([0.29])
        pt = detect_perforation(frame, film_format="8mm")
        assert pt is not None
        # Anchor should be near bottom edge (340)
        assert abs(pt[1] - FRAME_H * 0.34) < FRAME_H * 0.05

    def test_8mm_single_perf_fallback_returns_corner(self):
        """Only one perf visible → its bottom-right corner is returned, not None."""
        frame = make_frame([0.25])
        pt = detect_perforation(frame, film_format="8mm")
        assert pt is not None
        # Bottom edge of perf at 0.25 = 250 + 50 = 300
        assert abs(pt[1] - FRAME_H * 0.30) < FRAME_H * 0.08

    def test_8mm_no_perfs_returns_none(self):
        """All-black frame → None for 8mm format."""
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        assert detect_perforation(frame, film_format="8mm") is None

    def test_super8_detects_single_perf(self):
        """super8 path is unaffected by 8mm changes and still detects one perf."""
        frame = make_frame([0.25])
        pt = detect_perforation(frame, film_format="super8")
        assert pt is not None

    def test_super16_two_perfs_detected(self):
        """super16 uses the same 2-perf path as 8mm and returns bottom-right corner."""
        frame = make_frame([0.25, 0.75])
        pt = detect_perforation(frame, film_format="super16")
        assert pt is not None
        # Bottom edge of top perf at 0.25 = 250 + 50 = 300
        assert abs(pt[1] - FRAME_H * 0.30) < FRAME_H * 0.08


# ── Unit 3: debug frame generation ───────────────────────────────────────────

def _write_frame(path, frame):
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


class TestDebugFrames:

    def test_debug_jpeg_created_for_failed_frame(self):
        """Failed detection (all-black frame) produces a _debug.jpg."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out, \
             tempfile.TemporaryDirectory() as dbg:

            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "frame_001.jpg"), good)

            bad = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            _write_frame(os.path.join(inp, "frame_002.jpg"), bad)

            stabilize_folder(inp, out, debug_dir=dbg,
                             film_format="8mm", jpeg_quality=95)

            debug_files = os.listdir(dbg)
            assert any(
                "frame_002" in f and f.endswith("_debug.jpg")
                for f in debug_files
            ), f"Expected debug file for frame_002, got: {debug_files}"

    def test_no_debug_file_for_successful_detection(self):
        """Successful detection → no debug file written."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out, \
             tempfile.TemporaryDirectory() as dbg:

            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "frame_001.jpg"), good)

            stabilize_folder(inp, out, debug_dir=dbg,
                             film_format="super8", jpeg_quality=95)

            assert os.listdir(dbg) == [], "No debug files expected for a successful run"

    def test_debug_filename_uses_source_basename_with_suffix(self):
        """Debug file is named <source>_debug.jpg."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out, \
             tempfile.TemporaryDirectory() as dbg:

            bad = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            _write_frame(os.path.join(inp, "my_scan.jpg"), bad)
            # Add a second good frame so stabilize_folder does not raise
            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "good_frame.jpg"), good)

            stabilize_folder(inp, out, debug_dir=dbg,
                             film_format="8mm", jpeg_quality=95)

            assert "my_scan_debug.jpg" in os.listdir(dbg)

    def test_debug_jpeg_is_valid_and_roi_width(self):
        """Debug JPEG is readable and its width matches the ROI strip width."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out, \
             tempfile.TemporaryDirectory() as dbg:

            bad = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            _write_frame(os.path.join(inp, "frame.jpg"), bad)
            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "frame_good.jpg"), good)

            stabilize_folder(inp, out, debug_dir=dbg,
                             film_format="8mm", jpeg_quality=95)

            debug_path = os.path.join(dbg, "frame_debug.jpg")
            assert os.path.exists(debug_path)
            img = cv2.imread(debug_path)
            assert img is not None, "Debug JPEG is not a valid image"
            expected_roi_w = _roi_w(FRAME_W)
            assert img.shape[1] == expected_roi_w, (
                f"Expected debug width {expected_roi_w}, got {img.shape[1]}"
            )

    def test_no_debug_dir_does_not_raise(self):
        """debug_dir=None → no debug output, no errors."""
        with tempfile.TemporaryDirectory() as inp, \
             tempfile.TemporaryDirectory() as out:

            good = make_frame([0.25])
            _write_frame(os.path.join(inp, "frame.jpg"), good)
            # Should complete without raising even though no debug_dir is given
            stabilize_folder(inp, out, debug_dir=None,
                             film_format="super8", jpeg_quality=95)
