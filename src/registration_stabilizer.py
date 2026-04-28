import math
from dataclasses import dataclass

import cv2
import numpy as np

_MAX_ROTATION_RAD = math.radians(8.0)
_BREAK_ROTATION_RAD = math.radians(2.0)


@dataclass(frozen=True)
class RegistrationResult:
    ok: bool
    matrix: np.ndarray
    confidence: float
    method: str
    reason: str = ""

    @property
    def tx(self):
        return float(self.matrix[0, 2])

    @property
    def ty(self):
        return float(self.matrix[1, 2])

    @property
    def theta(self):
        return float(math.atan2(self.matrix[1, 0], self.matrix[0, 0]))


def identity_matrix():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def affine_from_components(tx, ty, theta_rad):
    c = float(math.cos(theta_rad))
    s = float(math.sin(theta_rad))
    return np.array([[c, -s, float(tx)], [s, c, float(ty)]], dtype=np.float32)


def components_from_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return (
        float(matrix[0, 2]),
        float(matrix[1, 2]),
        float(math.atan2(matrix[1, 0], matrix[0, 0])),
    )


def select_registration_roi(frame_shape, anchor, perf_bbox=None):
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("frame_shape must have positive height and width")

    cx = float(anchor[0])
    cy = float(anchor[1])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        raise ValueError("anchor coordinates must be finite")
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        raise ValueError("anchor coordinates must be inside the frame")

    if perf_bbox is not None:
        perf_w = max(1, int(perf_bbox[0]))
        half_w = max(80, int(perf_w * 2.5))
    else:
        half_w = max(80, int(w * 0.12))
    half_w = min(half_w, max(90, int(w * 0.10)))

    x0 = max(0, int(round(cx - half_w)))
    x1 = min(w, int(round(cx + half_w)))

    min_width = min(w, max(40, int(w * 0.08)))
    if x1 - x0 < min_width:
        center = int(round(np.clip(cx, 0, w - 1)))
        x0 = max(0, center - min_width // 2)
        x1 = min(w, x0 + min_width)
        x0 = max(0, x1 - min_width)

    return (int(x0), 0, int(x1), h)


def crop_roi(frame, roi):
    x0, y0, x1, y1 = roi
    return frame[int(y0) : int(y1), int(x0) : int(x1)]


def resize_for_registration(image, max_dim=900):
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image, 1.0

    scale = float(max_dim) / float(largest)
    resized = cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def scale_alignment_matrix(matrix, scale):
    matrix = np.asarray(matrix, dtype=np.float32).copy()
    if scale <= 0:
        raise ValueError("scale must be positive")
    matrix[0, 2] /= float(scale)
    matrix[1, 2] /= float(scale)
    return matrix


def local_matrix_to_frame(matrix, roi):
    matrix = np.asarray(matrix, dtype=np.float32)
    x0, y0, _, _ = roi
    x0 = float(x0)
    y0 = float(y0)

    full = matrix.copy()
    full[0, 2] = matrix[0, 2] + x0 - (matrix[0, 0] * x0 + matrix[0, 1] * y0)
    full[1, 2] = matrix[1, 2] + y0 - (matrix[1, 0] * x0 + matrix[1, 1] * y0)
    return full.astype(np.float32)


def preprocess_registration_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    gray = gray.astype(np.float32)
    if gray.size == 0:
        return gray

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (3, 3), 0)

    mean = float(np.mean(mag))
    std = float(np.std(mag))
    if std < 1e-6:
        return np.zeros_like(mag, dtype=np.float32)

    norm = (mag - mean) / std
    norm -= float(np.min(norm))
    max_val = float(np.max(norm))
    if max_val > 1e-6:
        norm /= max_val
    return norm.astype(np.float32)


def texture_score(processed_image):
    if processed_image.size == 0:
        return 0.0
    return float(np.std(processed_image))


def estimate_registration_transform(
    reference_processed,
    current_processed,
    *,
    min_texture=0.01,
    min_phase_response=0.04,
    min_ecc_confidence=0.35,
    max_translation=None,
    max_rotation_rad=_MAX_ROTATION_RAD,
):
    reference_processed = np.asarray(reference_processed, dtype=np.float32)
    current_processed = np.asarray(current_processed, dtype=np.float32)

    if reference_processed.shape != current_processed.shape:
        return RegistrationResult(
            False, identity_matrix(), 0.0, "failed", "shape mismatch"
        )

    if (
        texture_score(reference_processed) < min_texture
        or texture_score(current_processed) < min_texture
    ):
        return RegistrationResult(
            False, identity_matrix(), 0.0, "failed", "low texture"
        )

    h, w = reference_processed.shape[:2]
    if max_translation is None:
        max_translation = max(20.0, min(h, w) * 0.45)

    try:
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(
            reference_processed, current_processed, window
        )
        phase_dx = float(shift[0])
        phase_dy = float(shift[1])
        phase_response = float(response)
    except cv2.error as exc:
        return RegistrationResult(False, identity_matrix(), 0.0, "failed", str(exc))

    if not (
        np.isfinite(phase_dx) and np.isfinite(phase_dy) and np.isfinite(phase_response)
    ):
        return RegistrationResult(False, identity_matrix(), 0.0, "failed", "non-finite")

    if abs(phase_dx) > max_translation or abs(phase_dy) > max_translation:
        return RegistrationResult(
            False, identity_matrix(), phase_response, "failed", "translation too large"
        )

    ref_to_current = np.array(
        [[1.0, 0.0, phase_dx], [0.0, 1.0, phase_dy]], dtype=np.float32
    )

    try:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            60,
            1e-5,
        )
        ecc_confidence, ecc_matrix = cv2.findTransformECC(
            reference_processed,
            current_processed,
            ref_to_current.copy(),
            cv2.MOTION_EUCLIDEAN,
            criteria,
        )
        ecc_confidence = float(ecc_confidence)
        align_matrix = cv2.invertAffineTransform(ecc_matrix).astype(np.float32)
        tx, ty, theta = components_from_matrix(align_matrix)
        if (
            ecc_confidence >= min_ecc_confidence
            and abs(tx) <= max_translation
            and abs(ty) <= max_translation
            and abs(theta) <= max_rotation_rad
        ):
            return RegistrationResult(True, align_matrix, ecc_confidence, "ecc")
    except cv2.error:
        pass

    if phase_response < min_phase_response:
        return RegistrationResult(
            False, identity_matrix(), phase_response, "failed", "low phase response"
        )

    phase_align = affine_from_components(-phase_dx, -phase_dy, 0.0)
    return RegistrationResult(True, phase_align, phase_response, "phase")


def detect_transform_breaks(
    tx,
    ty,
    theta,
    *,
    valid_mask=None,
    translation_threshold=120.0,
    rotation_threshold=_BREAK_ROTATION_RAD,
):
    tx = np.asarray(tx, dtype=np.float32)
    ty = np.asarray(ty, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    n = len(tx)
    if valid_mask is None:
        valid_mask = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(theta)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    breaks = [0]
    last_valid_idx = None
    for i in range(n):
        if not valid_mask[i]:
            continue
        if last_valid_idx is None:
            last_valid_idx = i
            continue

        delta_t = math.hypot(
            float(tx[i] - tx[last_valid_idx]),
            float(ty[i] - ty[last_valid_idx]),
        )
        delta_r = abs(float(theta[i] - theta[last_valid_idx]))
        if delta_t > translation_threshold or delta_r > rotation_threshold:
            breaks.append(i)
        last_valid_idx = i
    return breaks


def smooth_transform_components(
    tx,
    ty,
    theta,
    *,
    radius=9,
    valid_mask=None,
    break_indices=None,
):
    tx = np.asarray(tx, dtype=np.float32)
    ty = np.asarray(ty, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    n = len(tx)
    if not (len(ty) == n and len(theta) == n):
        raise ValueError("tx, ty, and theta must have the same length")

    if valid_mask is None:
        valid_mask = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(theta)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    breaks = sorted(set(int(i) for i in (break_indices or [0]) if 0 <= int(i) < n))
    if not breaks or breaks[0] != 0:
        breaks.insert(0, 0)
    breaks.append(n)

    out = []
    for arr in (tx, ty, theta):
        smoothed = np.zeros(n, dtype=np.float32)
        for start, end in zip(breaks[:-1], breaks[1:], strict=True):
            segment = arr[start:end].astype(np.float32).copy()
            segment_valid = valid_mask[start:end] & np.isfinite(segment)
            filled = _fill_nans_1d(segment, segment_valid)
            smoothed[start:end] = _smooth_1d(filled, radius)
        out.append(smoothed)

    return tuple(out)


def _fill_nans_1d(values, valid_mask):
    values = values.astype(np.float32).copy()
    idx = np.arange(len(values))
    good = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if not len(values):
        return values
    if not np.any(good):
        return np.zeros_like(values, dtype=np.float32)
    values[~good] = np.interp(idx[~good], idx[good], values[good])
    return values


def _smooth_1d(values, radius):
    values = values.astype(np.float32)
    radius = int(radius or 0)
    if radius <= 0 or len(values) <= 1:
        return values.copy()

    radius = min(radius, max(1, len(values) - 1))
    kernel_size = radius * 2 + 1
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)
