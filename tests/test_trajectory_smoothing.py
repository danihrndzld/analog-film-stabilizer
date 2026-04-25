"""Tests for trajectory_smoothing: rigid fit, splice detection, smoothing."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trajectory_smoothing import (  # noqa: E402
    detect_splices,
    rigid_fit_2pt,
    smooth_trajectory,
)

# ── rigid_fit_2pt ────────────────────────────────────────────────────────────


class TestRigidFit:
    def test_pure_translation(self):
        ref = [(100.0, 200.0), (400.0, 200.0)]
        obs = [(105.5, 203.25), (405.5, 203.25)]
        tx, ty, theta = rigid_fit_2pt(ref, obs)
        assert tx == pytest.approx(5.5, abs=1e-6)
        assert ty == pytest.approx(3.25, abs=1e-6)
        assert theta == pytest.approx(0.0, abs=1e-9)

    def test_pure_rotation_about_centroid(self):
        # Rotate two points about their centroid by +0.1 rad.
        ref = np.array([[100.0, 200.0], [400.0, 200.0]])
        c = ref.mean(axis=0)
        angle = 0.1
        cs, sn = np.cos(angle), np.sin(angle)
        R = np.array([[cs, -sn], [sn, cs]])
        obs = (ref - c) @ R.T + c

        tx, ty, theta = rigid_fit_2pt(ref, obs)
        assert theta == pytest.approx(angle, abs=1e-9)
        # Applying fit to ref centroid should return obs centroid.
        rc = c
        rotated = np.array([cs * rc[0] - sn * rc[1], sn * rc[0] + cs * rc[1]])
        assert tx == pytest.approx(c[0] - rotated[0], abs=1e-6)
        assert ty == pytest.approx(c[1] - rotated[1], abs=1e-6)

    def test_compound_translation_and_rotation(self):
        ref = np.array([[50.0, 50.0], [250.0, 150.0]])
        angle = -0.15
        cs, sn = np.cos(angle), np.sin(angle)
        t_true = np.array([7.0, -3.5])
        obs = np.array(
            [
                [cs * p[0] - sn * p[1] + t_true[0], sn * p[0] + cs * p[1] + t_true[1]]
                for p in ref
            ]
        )
        tx, ty, theta = rigid_fit_2pt(ref, obs)
        assert theta == pytest.approx(angle, abs=1e-9)
        assert tx == pytest.approx(t_true[0], abs=1e-6)
        assert ty == pytest.approx(t_true[1], abs=1e-6)
        # Round-trip.
        for p, o in zip(ref, obs, strict=True):
            px = cs * p[0] - sn * p[1] + tx
            py = sn * p[0] + cs * p[1] + ty
            assert (px, py) == pytest.approx(tuple(o), abs=1e-6)

    def test_coincident_points_raises(self):
        with pytest.raises(ValueError):
            rigid_fit_2pt([(10.0, 10.0), (10.0, 10.0)], [(0.0, 0.0), (5.0, 5.0)])
        with pytest.raises(ValueError):
            rigid_fit_2pt([(0.0, 0.0), (5.0, 5.0)], [(1.0, 1.0), (1.0, 1.0)])


# ── smooth_trajectory ────────────────────────────────────────────────────────


class TestSmoothTrajectory:
    def test_clean_signal_reproduced(self):
        # Aggressive smoothing (window=51) attenuates higher frequencies by
        # design. Linear signals pass through without distortion. Ignore
        # savgol boundary zones of one window width.
        n = 800
        tx = np.linspace(0.0, 10.0, n)
        ty = np.linspace(0.0, -5.0, n)
        theta = np.linspace(0.0, 0.01, n)
        tx_s, ty_s, theta_s, mask = smooth_trajectory(tx, ty, theta, [0])
        w = 51
        assert np.max(np.abs(tx_s[w:-w] - tx[w:-w])) < 1e-6
        assert np.max(np.abs(ty_s[w:-w] - ty[w:-w])) < 1e-6
        assert np.max(np.abs(theta_s[w:-w] - theta[w:-w])) < 1e-9
        assert not mask.any()

    def test_nan_gaps_filled(self):
        n = 600
        tx_true = np.linspace(0.0, 10.0, n)
        ty_true = np.linspace(0.0, -5.0, n)
        tx = tx_true.copy()
        ty = ty_true.copy()
        theta = np.zeros(n)
        rng = np.random.default_rng(0)
        gap_idx = rng.choice(np.arange(10, n - 10), size=30, replace=False)
        tx[gap_idx] = np.nan
        ty[gap_idx] = np.nan
        theta[gap_idx] = np.nan

        tx_s, ty_s, theta_s, _ = smooth_trajectory(tx, ty, theta, [0])
        assert np.isfinite(tx_s).all()
        assert np.isfinite(ty_s).all()
        assert np.isfinite(theta_s).all()
        # Linear signal with sparse gaps: interior samples recovered exactly.
        w = 51
        assert np.max(np.abs(tx_s[w:-w] - tx_true[w:-w])) < 1e-6
        assert np.max(np.abs(ty_s[w:-w] - ty_true[w:-w])) < 1e-6

    def test_outlier_spike_flagged_and_replaced(self):
        n = 300
        rng = np.random.default_rng(42)
        tx = 10.0 + rng.normal(0.0, 0.1, n)
        ty = 5.0 + rng.normal(0.0, 0.1, n)
        theta = np.zeros(n)
        # Inject a blatant spike on tx at index 150.
        tx[150] = 1000.0

        tx_s, _, _, mask = smooth_trajectory(tx, ty, theta, [0])
        assert mask[150]
        # Replaced value should be close to the surrounding level.
        assert abs(tx_s[150] - 10.0) < 1.0
        # Neighbours away from the spike unaffected.
        assert abs(tx_s[100] - 10.0) < 1.0
        assert abs(tx_s[200] - 10.0) < 1.0

    def test_short_segment_fallback(self):
        # Segment shorter than polyorder+2 should fall back to median, not crash.
        tx = np.array([1.0, 2.0, 3.0])
        ty = np.array([4.0, 5.0, 6.0])
        theta = np.array([0.0, 0.0, 0.0])
        tx_s, ty_s, theta_s, mask = smooth_trajectory(tx, ty, theta, [0])
        assert tx_s.shape == (3,)
        assert np.isfinite(tx_s).all()
        assert not mask.any()

    def test_empty_input(self):
        tx_s, ty_s, theta_s, mask = smooth_trajectory([], [], [], [0])
        assert tx_s.shape == (0,)
        assert mask.shape == (0,)


# ── detect_splices ───────────────────────────────────────────────────────────


class TestDetectSplices:
    def test_step_in_translation(self):
        n = 400
        perf_spacing = 50.0
        tx = np.zeros(n)
        ty = np.zeros(n)
        theta = np.zeros(n)
        # 3× perf_spacing step at frame 200.
        tx[200:] += 3.0 * perf_spacing
        ncc = np.full(n, 0.9)

        splices = detect_splices(ncc, ncc, tx, ty, theta, perf_spacing)
        assert 0 in splices
        assert 200 in splices

    def test_gradual_drift_no_splice(self):
        n = 400
        perf_spacing = 50.0
        tx = np.linspace(0, 20, n)  # ~0.05 px/frame, well below threshold
        ty = np.zeros(n)
        theta = np.zeros(n)
        ncc = np.full(n, 0.9)

        splices = detect_splices(ncc, ncc, tx, ty, theta, perf_spacing)
        assert splices == [0]

    def test_ncc_collapse_alone(self):
        n = 200
        perf_spacing = 50.0
        tx = np.zeros(n)
        ty = np.zeros(n)
        theta = np.zeros(n)
        ncc_a1 = np.full(n, 0.9)
        ncc_a2 = np.full(n, 0.9)
        # Both anchors collapse on frame 120.
        ncc_a1[120] = 0.05
        ncc_a2[120] = 0.05

        splices = detect_splices(ncc_a1, ncc_a2, tx, ty, theta, perf_spacing)
        assert 120 in splices

    def test_min_segment_merges(self):
        n = 400
        perf_spacing = 50.0
        tx = np.zeros(n)
        ty = np.zeros(n)
        theta = np.zeros(n)
        # Two splices close together — second should be dropped when within
        # min_segment of the first.
        tx[100:] += 3.0 * perf_spacing
        tx[110:] += 3.0 * perf_spacing
        ncc = np.full(n, 0.9)

        splices = detect_splices(ncc, ncc, tx, ty, theta, perf_spacing, min_segment=30)
        # Only one of the two close boundaries should survive.
        near = [s for s in splices if 95 <= s <= 115]
        assert len(near) == 1

    def test_empty_input(self):
        assert detect_splices([], [], [], [], [], 50.0) == [0]


# ── integration ──────────────────────────────────────────────────────────────


class TestIntegration:
    def test_multi_segment_smoothing_no_cross_bleed(self):
        n = 500
        perf_spacing = 50.0
        tx = np.zeros(n)
        ty = np.zeros(n)
        theta = np.zeros(n)
        # Segment A: flat at 0. Segment B (from 250): flat at 500.
        tx[250:] = 500.0
        ncc = np.full(n, 0.9)

        splices = detect_splices(ncc, ncc, tx, ty, theta, perf_spacing)
        assert 250 in splices

        tx_s, _, _, _ = smooth_trajectory(tx, ty, theta, splices)
        # Values well inside segment A should still be ~0 (not pulled toward 500).
        assert abs(tx_s[100]) < 1.0
        # Values well inside segment B should still be ~500.
        assert abs(tx_s[400] - 500.0) < 1.0
