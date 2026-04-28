---
title: "fix: Registration-first stabilization"
type: fix
status: active
date: 2026-04-27
---

# fix: Registration-first stabilization

## Overview

Replace the current point-tracking-first stabilization path with a registration-first path. The user's click still selects a reference region, but the backend now estimates each frame's motion from a tall film-edge/perforation strip using edge-space image registration, smooths the resulting full-frame transform trajectory, and warps frames from that global trajectory. The existing NCC anchor tracker remains as a fallback/debug signal, not the primary source of motion.

This is intentionally different from the prior single-perforation approach: one chosen point should not decide the whole frame. A long static strip carries more evidence, rejects one-off false matches, and gives the smoother a physically meaningful trajectory instead of a sequence of independently chosen anchor centers.

---

## Problem Frame

User feedback: "no está funcionando nada bien la estabilización, se mueven mucho entre sí las fotos" and "pensemos algo radicalmente diferente." In the current code, `src/perforation_stabilizer_app.py` builds one template from one anchor, re-detects that anchor in every frame, fills failures by interpolation, then applies translation-only warps. That creates two structural problems:

- Per-frame NCC decisions can jump between similar perforations or bright shapes; any accepted wrong point becomes visible frame jitter.
- The stabilization transform is inferred from a single point, so it cannot reliably separate translation, small rotation, or tracker noise.

Prior documents already explored stronger anchor tracking and two-anchor smoothing, but the checked-out code still uses the single-anchor CLI/UI contract. This plan takes a backend-first route: keep the one-click workflow and replace the primary motion estimator with strip registration.

---

## Requirements Trace

- R1. Preserve the existing one-click UI/CLI contract so the change is immediately usable from Electron and packaged binaries.
- R2. Estimate per-frame motion from a tall registration strip around the clicked film/perforation region, not from a single NCC point.
- R3. Use edge/gradient-space preprocessing so exposure and density drift affect registration less than raw grayscale NCC.
- R4. Prefer Euclidean registration (translation + small rotation) and fall back to translation-only phase correlation or previous transform when confidence is poor.
- R5. Smooth the full transform trajectory before warping, using `smooth_radius` as an active control rather than a dead parameter.
- R6. Preserve existing output behavior: same output folder shape, progress callbacks, JPEG/PNG behavior, border modes, and report file.
- R7. Extend the report with registration metrics that make failures diagnosable.
- R8. Keep the old anchor tracker available as fallback/debug support so low-texture strip failures do not make the pipeline unusable.

---

## Scope Boundaries

- No two-anchor UI migration in this change.
- No machine-learning model, training set, or external service.
- No automatic film-format classifier.
- No major Electron redesign; copy changes are allowed where they clarify the click target.
- No packaged DMG release in this implementation unit unless the user explicitly asks after verification.
- No guarantee that scene-content registration is used; the registration ROI is deliberately biased toward the static film/perforation side, not the moving picture area.

---

## Context & Research

### Relevant Code and Patterns

- `src/perforation_stabilizer_app.py` currently owns the full backend pipeline, including `list_images`, `_build_perforation_template`, `_template_match_candidates`, `_rank_candidates`, `_MotionPredictor`, `_detect_perf_spacing`, and `stabilize_folder`.
- `src/stabilizer_cli.py` forwards `--anchor-x` and `--anchor-y` into `stabilize_folder`; preserving this contract keeps Electron and existing manual smoke commands working.
- `electron/main.js` passes the single anchor coordinates to the Python CLI in packaged and dev modes.
- `electron/renderer/renderer.js` and `electron/renderer/index.html` describe the clicked point as a "referencia"; copy can be tightened to "zona de perforación/borde" without changing interaction structure.
- `tests/test_detection.py` already has synthetic frame helpers and workflow tests for `stabilize_folder`, making it the right place for backend regression coverage.

### Institutional Learnings

- `docs/solutions/best-practices/post-refactor-review-checklist-dead-code-stale-references-2026-04-14.md`: after replacing a detection/alignment subsystem, audit stale parameters and cross-process numeric values. This directly applies because `smooth_radius` is currently accepted but not used for smoothing the final motion path.
- `docs/ideation/2026-04-23-stabilization-quality-rethink-ideation.md`: gradient/edge-space NCC and using more global trajectory evidence were identified as high-leverage quality improvements. This plan chooses a stronger variant: registration over a strip, with the point tracker demoted to fallback.

### External References

- None. The plan uses OpenCV primitives already present in the dependency set: `cv2.Sobel`, `cv2.phaseCorrelate`, `cv2.findTransformECC`, and `cv2.warpAffine`.

---

## Key Technical Decisions

- **Registration strip over point tracking:** Build a tall ROI centered on the clicked anchor's X coordinate and spanning most or all of the frame height. This captures the perforation column, film edge, dust/edge texture, and repeated sprocket geometry as a single registration surface.
- **Edge-space preprocessing:** Convert the registration strip to gradient magnitude with local normalization before registration. Perforation and film boundaries are more stable than raw luminance under flicker, scanner exposure changes, and emulsion density shifts.
- **Euclidean-first, phase fallback:** Try `cv2.findTransformECC` with `MOTION_EUCLIDEAN` initialized from the previous accepted transform. When ECC fails or returns a low-quality transform, fall back to translation-only phase correlation. When both fail, reuse the previous smoothed-compatible transform and mark the frame as fallback.
- **Smooth transforms, not detected points:** Convert per-frame raw transforms to `(tx, ty, theta)` arrays, robust-fill missing values, segment at large transform discontinuities or long reuse runs, smooth each component with the existing `smooth_radius`, then warp from the smoothed trajectory.
- **Anchor tracker retained as safety net:** The existing NCC point tracker stays available for debug frames and as a last-resort translation estimate if strip registration cannot find a usable signal.
- **No CLI break:** Keep `--anchor-x/--anchor-y`; this is a stabilization quality fix, not a UX migration.

---

## Open Questions

### Resolved During Planning

- **Should this implement the prior two-anchor plan?** No. The user asked for a radical rethink after bad stabilization, and the current checkout does not contain the two-anchor architecture. Strip registration is more radical while staying implementable behind the existing single-anchor contract.
- **Should registration use the full frame?** No. Full-frame registration can lock to moving scene content. The ROI should favor the static film/perforation region around the user's click.
- **Should `smooth_radius` stay user-visible?** Yes. It already exists in the UI and CLI; this plan makes it real again by applying it to the transform trajectory.

### Deferred to Implementation

- Exact strip width multiplier. Start from detected perforation width when available, otherwise use a frame-width heuristic; tune only enough to pass synthetic and smoke verification.
- Exact ECC confidence threshold. Use ECC's returned correlation coefficient plus low-texture checks and transform sanity checks, and keep conservative fallback behavior.
- Whether the fallback tracker is called for every frame or only when registration fails. Prefer only-on-failure unless tests reveal the extra signal is needed for report/debug quality; initialize its template, predictor, and perf-spacing state before the registration loop either way.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```text
Reference frame + user click
  -> detect perforation bbox when possible
  -> choose a tall registration strip around the clicked film/perf region
  -> preprocess strip into normalized edge magnitude

For each frame
  -> crop same strip bounds
  -> preprocess to edge magnitude
  -> estimate current->reference transform:
       1. ECC Euclidean registration, initialized from previous transform
       2. phase correlation translation fallback
       3. legacy anchor-tracker translation fallback
       4. previous transform reuse if all else fails
  -> record tx, ty, theta, method, confidence

Global trajectory pass
  -> split at large discontinuities or long fallback/reuse runs
  -> fill missing transform values from nearby valid frames within each segment
  -> reject implausible spikes by delta/MAD checks
  -> smooth tx, ty, theta within each segment using smooth_radius

Warp pass
  -> apply smoothed inverse/current->reference transform to each full frame
  -> write frames and stabilization_report.txt
```

---

## Implementation Units

- U1. **Registration primitives**

**Goal:** Add pure helper functions for choosing a registration strip, preprocessing it into edge-space, estimating a frame transform, smoothing transform trajectories, and building warp matrices.

**Requirements:** R2, R3, R4, R5.

**Dependencies:** None.

**Files:**
- Create: `src/registration_stabilizer.py`
- Test: `tests/test_registration_stabilizer.py`

**Approach:**
- Represent transform trajectories as three arrays: `tx`, `ty`, and `theta_rad`.
- Add ROI selection that uses `_detect_perf_bbox` output when available and otherwise falls back to a stable frame-size heuristic around the clicked X coordinate.
- Preprocess strips with grayscale conversion, mild blur, Sobel/Scharr gradient magnitude, and normalization to `float32`.
- Estimate Euclidean registration with ECC where possible; sanity-check finite values, reasonable translation magnitude, and small rotation.
- Add phase-correlation translation fallback using the same preprocessed strips.
- Smooth arrays using segmented NaN fill plus a centered moving-average/median-style filter driven by `smooth_radius`; segmentation should prevent one large discontinuity from bleeding into neighboring frames.

**Patterns to follow:**
- Keep helpers small and testable, matching the existing pure-function style in `src/perforation_stabilizer_app.py`.
- Avoid SciPy; the project dependencies are currently NumPy and OpenCV only.

**Test scenarios:**
- Happy path: translated synthetic strip returns a transform whose inverse aligns the strip within a small pixel tolerance.
- Happy path: small rotated synthetic strip returns finite `theta_rad` and does not collapse to a huge translation.
- Edge case: blank/low-texture strips return a failure result instead of a bogus high-confidence transform.
- Edge case: NaN gaps in a transform trajectory are filled and smoothed without producing NaNs.
- Edge case: a one-frame translation spike is damped by smoothing while neighboring values remain close to the baseline.
- Edge case: a large persistent step creates separate smoothing segments so pre-step frames are not pulled toward post-step frames.
- Integration: ROI selection near the left edge clips to frame bounds and still returns a non-empty strip.

**Verification:**
- Unit tests prove transform sign convention by warping a shifted synthetic frame back to reference coordinates.

---

- U2. **Backend pipeline integration**

**Goal:** Rework `stabilize_folder` so registration-first transforms drive the final warp, while preserving the public function signature and output contract.

**Requirements:** R1, R2, R4, R5, R6, R7, R8.

**Dependencies:** U1.

**Files:**
- Modify: `src/perforation_stabilizer_app.py`
- Test: `tests/test_detection.py`

**Approach:**
- Keep `stabilize_folder(..., anchor, smooth_radius, ...)` as-is.
- Build the reference registration strip from the first readable frame and the clicked anchor.
- Initialize the legacy anchor fallback before pass 1: template, perf spacing, search radius, and `_MotionPredictor`.
- In pass 1, estimate a raw transform for every frame using U1 helpers. Initialize each frame from the previous accepted transform.
- On registration failure, run the existing anchor tracker path as a fallback translation estimate without contaminating its predictor on ambiguous matches. If that also fails, reuse the previous transform and increment fallback counters.
- Smooth the raw transform trajectory after pass 1.
- In pass 2, warp each full frame using the smoothed transform matrix rather than translation from a detected point.
- Keep progress callbacks split across detection/registration and writing.
- Write a report with existing keys plus `registration_success_frames`, `phase_fallback_frames`, `anchor_fallback_frames`, `reused_transform_frames`, `mean_registration_confidence`, `registration_roi`, and `smooth_radius`.

**Patterns to follow:**
- Preserve the existing log/progress/report style in `stabilize_folder`.
- Reuse `_build_perforation_template`, `_template_match_candidates`, `_rank_candidates`, `_MotionPredictor`, and `_detect_perf_spacing` for fallback instead of deleting the existing tracker.

**Test scenarios:**
- Happy path: identical synthetic frames still produce the expected number of output images and zero failures.
- Happy path: frames shifted by known `(dx, dy)` are warped back so the perforation centroid aligns to the reference within tolerance.
- Happy path: `smooth_radius` changes the reported value and affects the trajectory smoother on a jittery synthetic sequence.
- Error path: `anchor=None` still raises `ValueError`.
- Error path: empty input folder still raises `RuntimeError`.
- Edge case: one blank frame in the middle does not crash; output count remains complete and report records fallback/reuse.
- Integration: `stabilization_report.txt` contains both legacy and registration metrics.

**Verification:**
- Existing `TestAnchorWorkflow` and debug-frame tests remain green after updating assertions for registration metrics.

---

- U3. **CLI and Electron copy alignment**

**Goal:** Keep process contracts stable while making the UI/CLI language match the new registration behavior.

**Requirements:** R1, R6.

**Dependencies:** U2.

**Files:**
- Modify: `src/stabilizer_cli.py`
- Modify: `electron/renderer/index.html`
- Modify: `electron/renderer/renderer.js`

**Approach:**
- Keep `--anchor-x` and `--anchor-y`.
- Update CLI help text from "Reference anchor" to "reference film/perforation region".
- Update preview copy so the click target is the film/perforation edge region, not an arbitrary point in the picture area.
- Leave `electron/main.js` argument forwarding unchanged unless implementation reveals stale validation or logging.

**Patterns to follow:**
- Keep Spanish UI copy concise and consistent with existing labels.

**Test scenarios:**
- Happy path: `src/stabilizer_cli.py --help` still shows `--anchor-x`/`--anchor-y`, with updated descriptions.
- Manual UI smoke: selecting a folder still enables the run button only after a preview click.

**Verification:**
- No cross-process contract changes are required for packaged or dev Electron launches.

---

- U4. **Regression and real-batch smoke verification**

**Goal:** Verify that the new path is mechanically correct and produces a materially steadier output on the available local batch.

**Requirements:** R5, R6, R7.

**Dependencies:** U1, U2, U3.

**Files:**
- Modify: `tests/test_detection.py`
- Create or update only if useful: a focused test fixture/helper inside `tests/test_registration_stabilizer.py`

**Approach:**
- Add synthetic jitter tests that measure alignment after stabilization rather than only output counts.
- Run the local test suite.
- Run a small smoke over a subset or the full `EXPORT TEST FRAMES/` folder using known prior anchor coordinates if available from recent commands or docs.
- Inspect the report counters to confirm registration, fallback, and smoothing are actually active.

**Patterns to follow:**
- Keep large real media out of tests; real-batch smoke is verification, not committed fixture data.

**Test scenarios:**
- Integration: a short translated sequence produces stabilized output with lower centroid variance than the input sequence.
- Integration: a sequence with one registration failure still writes every frame and records a fallback metric.
- Regression: existing template-building, top-K, ranking, and debug tests still pass.

**Verification:**
- `uv run pytest` passes.
- A smoke run writes output frames and a report with non-zero registration metrics.

---

## System-Wide Impact

- **Interaction graph:** Electron renderer collects one click, Electron main forwards it unchanged, CLI passes it unchanged, backend interprets it as a registration-region seed.
- **Error propagation:** Registration failures should degrade to fallback counters and previous-transform reuse when possible; only total inability to build a reference strip or read frames should abort the run.
- **State lifecycle risks:** The previous transform can bias later frames if reused for too long. The report must surface reuse counts so this failure is visible.
- **API surface parity:** Public CLI arguments remain compatible. `stabilize_folder` signature remains compatible.
- **Integration coverage:** Synthetic output-alignment tests must verify actual stabilization behavior, not just output file creation.
- **Unchanged invariants:** Output directory layout, file naming, border modes, JPEG quality semantics, preview mode, and JSON-lines protocol remain unchanged.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Registration locks to moving picture content instead of the static film edge | ROI selection must be biased to the clicked perforation/edge strip, not the full frame |
| ECC fails on low-texture or overexposed strips | Phase-correlation fallback, anchor-tracker fallback, and previous-transform reuse keep the run complete |
| Transform sign convention is wrong | U1 tests must warp shifted synthetic frames back to reference coordinates |
| Smoothing hides true splice/discontinuity | Keep smoothing local and conservative in this first pass; report fallback/reuse counts for suspicious segments |
| Existing tests assume point-tracker metrics | Update tests to assert preserved behavior plus new registration metrics, not stale internals |

---

## Documentation / Operational Notes

- The user-facing workflow remains one click, but the copy should nudge users to click the perforation/film-edge area because that region now seeds the registration strip.
- The report becomes the primary debug artifact for deciding whether a run relied on ECC, phase fallback, anchor fallback, or transform reuse.
- After code verification, a release pass should rebuild the packaged binary and DMG separately; that is outside this plan's immediate implementation scope.

---

## Sources & References

- Related ideation: `docs/ideation/2026-04-23-stabilization-quality-rethink-ideation.md`
- Related requirements: `docs/brainstorms/2026-04-18-robust-perforation-tracking-requirements.md`
- Related prior plan: `docs/plans/2026-04-19-002-feat-multi-anchor-and-global-trajectory-smoothing-plan.md`
- Related learning: `docs/solutions/best-practices/post-refactor-review-checklist-dead-code-stale-references-2026-04-14.md`
- Backend integration: `src/perforation_stabilizer_app.py`
- CLI integration: `src/stabilizer_cli.py`
- Electron integration: `electron/main.js`, `electron/renderer/renderer.js`, `electron/renderer/index.html`
