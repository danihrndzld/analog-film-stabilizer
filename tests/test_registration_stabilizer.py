import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from registration_stabilizer import (
    crop_roi,
    detect_transform_breaks,
    estimate_registration_transform,
    local_matrix_to_frame,
    preprocess_registration_image,
    resize_for_registration,
    scale_alignment_matrix,
    select_registration_roi,
    smooth_transform_components,
)


def make_registration_frame(width=240, height=320):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (18, 0), (34, height), (45, 45, 45), -1)
    for y in range(45, height, 95):
        cv2.rectangle(frame, (50, y), (95, y + 42), (235, 235, 235), -1)
        cv2.line(frame, (45, y - 18), (115, y - 8), (120, 120, 120), 2)
        cv2.circle(frame, (115, y + 24), 7, (150, 150, 150), -1)
    cv2.line(frame, (125, 0), (125, height), (80, 80, 80), 2)
    return frame


def test_select_registration_roi_clips_near_left_edge():
    roi = select_registration_roi((320, 240, 3), (55.0, 80.0), perf_bbox=(45, 42))

    x0, y0, x1, y1 = roi
    assert x0 == 0
    assert y0 == 0
    assert x1 > x0
    assert y1 == 320


def test_select_registration_roi_rejects_bad_anchor_y():
    try:
        select_registration_roi((320, 240, 3), (55.0, float("nan")))
    except ValueError as exc:
        assert "anchor coordinates" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-finite anchor y")


def test_resize_and_local_matrix_scale_back_to_frame_coordinates():
    image = make_registration_frame(width=300, height=600)
    resized, scale = resize_for_registration(image, max_dim=150)
    assert resized.shape[0] == 150
    assert 0 < scale < 1

    scaled_matrix = np.float32([[1, 0, -3 * scale], [0, 1, 5 * scale]])
    local_matrix = scale_alignment_matrix(scaled_matrix, scale)
    full_matrix = local_matrix_to_frame(local_matrix, (20, 10, 220, 610))

    assert abs(full_matrix[0, 2] + 3) < 1e-5
    assert abs(full_matrix[1, 2] - 5) < 1e-5


def test_phase_or_ecc_transform_aligns_translated_strip():
    reference = make_registration_frame()
    shifted = cv2.warpAffine(
        reference,
        np.float32([[1, 0, 7], [0, 1, -4]]),
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
    )
    roi = select_registration_roi(reference.shape, (70.0, 120.0), perf_bbox=(45, 42))

    ref_processed = preprocess_registration_image(crop_roi(reference, roi))
    cur_processed = preprocess_registration_image(crop_roi(shifted, roi))
    result = estimate_registration_transform(ref_processed, cur_processed)

    assert result.ok, result.reason
    assert result.method in {"ecc", "phase"}
    assert abs(result.tx + 7) < 1.5
    assert abs(result.ty - 4) < 1.5

    aligned = cv2.warpAffine(
        shifted,
        result.matrix,
        (shifted.shape[1], shifted.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    aligned_processed = preprocess_registration_image(crop_roi(aligned, roi))
    assert np.mean(np.abs(aligned_processed - ref_processed)) < 0.08


def test_phase_fallback_has_correct_translation_sign_when_ecc_rejected():
    reference = make_registration_frame()
    shifted = cv2.warpAffine(
        reference,
        np.float32([[1, 0, 7], [0, 1, -4]]),
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
    )
    roi = select_registration_roi(reference.shape, (70.0, 120.0), perf_bbox=(45, 42))

    result = estimate_registration_transform(
        preprocess_registration_image(crop_roi(reference, roi)),
        preprocess_registration_image(crop_roi(shifted, roi)),
        min_ecc_confidence=2.0,
    )

    assert result.ok, result.reason
    assert result.method == "phase"
    assert abs(result.tx + 7) < 1.5
    assert abs(result.ty - 4) < 1.5


def test_rotation_estimate_stays_finite_for_small_rotation():
    reference = make_registration_frame()
    h, w = reference.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 1.0, 1.0)
    rotated = cv2.warpAffine(reference, rot, (w, h), borderMode=cv2.BORDER_CONSTANT)
    roi = select_registration_roi(reference.shape, (70.0, 120.0), perf_bbox=(45, 42))

    result = estimate_registration_transform(
        preprocess_registration_image(crop_roi(reference, roi)),
        preprocess_registration_image(crop_roi(rotated, roi)),
        min_ecc_confidence=0.2,
    )

    assert result.ok, result.reason
    assert result.method == "ecc"
    assert np.isfinite(result.theta)
    assert 0.2 < abs(np.rad2deg(result.theta)) < 3.0


def test_blank_strip_returns_failure_instead_of_identity_success():
    blank = np.zeros((240, 120), dtype=np.float32)

    result = estimate_registration_transform(blank, blank)

    assert not result.ok
    assert result.reason == "low texture"


def test_smoothing_fills_gaps_and_damps_single_frame_spike():
    tx = np.array([0, 0, np.nan, 30, 0, 0], dtype=np.float32)
    ty = np.zeros_like(tx)
    theta = np.zeros_like(tx)
    valid = np.isfinite(tx)

    sx, sy, st = smooth_transform_components(tx, ty, theta, radius=1, valid_mask=valid)

    assert np.all(np.isfinite(sx))
    assert np.all(np.isfinite(sy))
    assert np.all(np.isfinite(st))
    assert sx[3] < 20


def test_large_persistent_step_creates_smoothing_segments():
    tx = np.array([0, 1, 2, 100, 101, 102], dtype=np.float32)
    ty = np.zeros_like(tx)
    theta = np.zeros_like(tx)

    breaks = detect_transform_breaks(tx, ty, theta, translation_threshold=40.0)
    sx, _, _ = smooth_transform_components(
        tx, ty, theta, radius=2, break_indices=breaks
    )

    assert breaks == [0, 3]
    assert sx[2] < 10
    assert sx[3] > 90


def test_invalid_gap_compares_next_valid_transform_to_previous_valid():
    tx = np.array([0, 0, 0, 0, 100], dtype=np.float32)
    ty = np.zeros_like(tx)
    theta = np.zeros_like(tx)
    valid = np.array([True, True, False, False, True])

    breaks = detect_transform_breaks(
        tx, ty, theta, valid_mask=valid, translation_threshold=40.0
    )

    assert breaks == [0, 4]
