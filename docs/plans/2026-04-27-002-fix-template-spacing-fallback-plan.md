---
title: "fix: Recover spacing when contour calibration fails"
type: fix
status: active
date: 2026-04-27
origin: docs/plans/2026-04-23-001-feat-stabilization-quality-pass-plan.md
---

# fix: Recover spacing when contour calibration fails

## Overview

The packaged app now reaches the backend, but Diego's `EXPORT TEST FRAMES/` batch still aborts before tracking:

- Phase 0 calibration samples 30 frames and gets 0 useful measurements.
- Default mode correctly falls back to first-frame bootstrap.
- Bootstrap then calls the same contour-based perforation spacing detector and aborts when it cannot find ≥3 bright contours in the first-frame strip.

This fix keeps the current two-anchor workflow but makes spacing estimation more robust: when contour spacing fails, derive spacing from template-matched repeated perforations and, as a last resort, from the two user anchors when they are vertically separated perforations.

---

## Problem Frame

`_detect_perf_spacing()` is contour-based and requires at least 3 perf-like bright contours in a narrow strip. That is too strict for overexposed or low-contrast frames where template matching can still track the user-selected perforation patch. The quality-pass plan promised default calibration failure should degrade to a warning and still produce output when possible. The current fallback path violates that intent by aborting on the same brittle spacing detector.

---

## Requirements Trace

- R1. Default mode must not abort solely because contour-based spacing detection fails when usable anchor templates exist.
- R2. Calibration should count sampled frames as useful when spacing can be recovered by template-based repeated-perf detection.
- R3. First-frame bootstrap should recover spacing with the same helper before raising the current "no pude detectar el espaciado" error.
- R4. The fallback must remain bounded: invalid anchors, off-frame anchors, or truly untrackable templates still raise clear errors.
- R5. Existing report fields and calibration status semantics remain unchanged: successful recovered calibration reports `ok`; fallback reports `fallback`.
- R6. The rebuilt packaged arm64 app must accept the two-anchor args and progress past this spacing failure on the test batch.

---

## Scope Boundaries

- Do not redesign the UI or add new user controls.
- Do not change CLI argument names.
- Do not tune smoothing, splice detection, consensus gates, or output crop behavior.
- Do not reintroduce old automatic format selectors or contour detector flows.
- Do not promise visual stabilization quality in this fix; the gate is recovering spacing well enough to start tracking.

---

## Context & Research

### Relevant Code and Patterns

- `src/perforation_stabilizer_app.py` contains `_detect_perf_spacing()`, `_template_match_candidates()`, and the `stabilize_folder()` calibration/bootstrap handoff.
- `src/calibration.py` currently calls `_detect_perf_spacing(frame, anchor1)`, then anchor2, before measuring NCC. If spacing fails, the sample is skipped entirely.
- `tests/test_calibration.py` covers calibration success, fallback, strict abort, and logging.
- `tests/test_detection.py` covers spacing detector behavior, two-anchor summary fields, anchor workflow, and health checks.
- `docs/plans/2026-04-23-001-feat-stabilization-quality-pass-plan.md` defines R2 fallback as warning + single-frame bootstrap in default mode, with strict abort only when opted in.

### Institutional Learnings

- `docs/solutions/best-practices/post-refactor-review-checklist-dead-code-stale-references-2026-04-14.md` highlights Electron-to-Python failures and silent fallback mismatches as review targets after cross-layer refactors.

### External References

- None. This is local OpenCV/template-matching behavior.

---

## Key Technical Decisions

- Add a small helper that tries spacing strategies in order: contour spacing around anchor 1, contour spacing around anchor 2, template-based repeated-perf spacing, then anchor-pair spacing inference.
- Keep `_detect_perf_spacing()` intact for existing contour behavior; add fallback helpers rather than weakening its current filters.
- Template-based spacing should reuse `_template_match_candidates()` with the already-built anchor templates, so it follows the same matching surface used by actual tracking.
- Anchor-pair inference is last-resort and only valid when the anchors are mostly vertically separated and the inferred interval count yields a plausible spacing relative to the detected template/perf height.
- Logging should say which fallback recovered spacing so user-facing diagnostics explain why calibration is degraded or recovered.

---

## Open Questions

### Resolved During Planning

- **Root cause class:** spacing-estimation brittleness, not the packaged CLI arg mismatch.
- **Compatibility direction:** retain strict two-anchor CLI and current UI.
- **Primary fallback:** template matching, because the selected anchors already define the patch the tracker will use.

### Deferred to Implementation

- Exact template-matching confidence floor and candidate count for spacing recovery should be chosen from existing `_template_match_candidates()` defaults and adjusted only enough for synthetic tests plus the local test batch.
- Exact anchor-pair interval-count heuristic should be conservative and covered by tests.

---

## Implementation Units

- U1. **Add spacing recovery helper**

**Goal:** Centralize robust spacing estimation so calibration and bootstrap do not duplicate brittle fallback logic.

**Requirements:** R1, R2, R3, R4

**Dependencies:** None

**Files:**
- Modify: `src/perforation_stabilizer_app.py`
- Test: `tests/test_detection.py`

**Approach:**
- Add a helper such as `_recover_perf_spacing(frame, anchor1, anchor2, template1=None, anchor1_in_tpl=None, template2=None, anchor2_in_tpl=None, log=None)`.
- The helper first delegates to `_detect_perf_spacing()` for both anchors.
- If templates are present, run vertical template matching around one or both anchors, gather repeated candidate Y coordinates, and return the median adjacent spacing when at least 2 plausible intervals agree.
- If template matching still cannot derive spacing, infer from the anchor pair when the anchors are mostly vertically separated; divide the Y distance by a plausible integer interval count and reject implausibly tight spacings.
- Return `None` only when no strategy can produce a bounded positive spacing.

**Patterns to follow:**
- Existing `_detect_perf_spacing()` returns `None` instead of raising.
- Existing template matching helpers operate on grayscale input and return candidate tuples.

**Test scenarios:**
- Happy path: existing contour spacing still returns the same value for synthetic three-perf frames.
- Recovery path: monkeypatch contour spacing to return `None`; repeated template candidates at Y positions 100, 400, 700 recover spacing near 300.
- Recovery path: when template candidates are insufficient, anchors separated by 900px over 3 intervals recover spacing near 300.
- Error path: non-finite anchors or implausible anchor-pair geometry return `None`.

**Verification:**
- Focused detection tests prove all spacing strategy branches.

---

- U2. **Use recovered spacing in calibration and bootstrap**

**Goal:** Make Phase 0 calibration and R2 fallback share the spacing recovery helper.

**Requirements:** R1, R2, R3, R5

**Dependencies:** U1

**Files:**
- Modify: `src/calibration.py`
- Modify: `src/perforation_stabilizer_app.py`
- Test: `tests/test_calibration.py`

**Approach:**
- In `run_calibration()`, after templates are built, replace direct `_detect_perf_spacing()` calls with the shared recovery helper using the calibration templates.
- In `stabilize_folder()` fallback mode, after first-frame templates are built, call the same helper before raising the bootstrap spacing error.
- Preserve strict-calibration behavior: strict mode still aborts when calibration returns `None`; the helper simply makes fewer samples fail spuriously.
- Add user-facing log lines only when fallback recovery, not primary contour detection, supplies spacing.

**Patterns to follow:**
- Existing `run_calibration(..., log=log)` logging style.
- Existing fallback summary fields in `stabilize_folder()`.

**Test scenarios:**
- Calibration path: synthetic batch where contour spacing is forced to fail but template spacing succeeds returns `CalibrationState`.
- Fallback path: small synthetic batch with forced contour failure but anchor-pair/template spacing recovery still produces output and `calibration_status == "fallback"`.
- Strict mode: calibration failure with no recovery still raises strict calibration error.
- Error path: no spacing recovery still raises the existing bootstrap spacing error.

**Verification:**
- `tests/test_calibration.py` and relevant `tests/test_detection.py` cases pass.

---

- U3. **Rebuild and smoke-test packaged app**

**Goal:** Put the recovered spacing logic into the local packaged app and confirm this specific error no longer appears.

**Requirements:** R6

**Dependencies:** U1, U2

**Files:**
- Generated only: `dist-py/stabilizer_arm64`, `dist/mac-arm64/Perforation Stabilizer.app`

**Approach:**
- Run the local build script, which rebuilds the arm64 PyInstaller backend and packages the Electron app.
- Verify packaged backend help still exposes the two-anchor CLI contract.
- Open the rebuilt app and retry `EXPORT TEST FRAMES/`.

**Patterns to follow:**
- Existing local packaging flow from `scripts/build-local.sh`.

**Test scenarios:**
- Integration: the same folder/anchor workflow no longer stops at "no pude detectar el espaciado entre perforaciones".
- Integration: progress advances into tracking or surfaces a later, different actionable error.

**Verification:**
- Rebuilt app is running and the previous spacing error is gone.

---

## System-Wide Impact

- **Interaction graph:** Electron sends anchors to CLI; CLI calls `stabilize_folder`; `stabilize_folder` calls calibration and fallback; both now share spacing recovery.
- **Error propagation:** unrecoverable spacing failures still emit the existing Spanish error; recoverable spacing failures become log warnings and continue.
- **State lifecycle risks:** generated binaries remain ignored; source changes are tests/backend only.
- **API surface parity:** dev-mode Python source and packaged PyInstaller backend use the same recovered spacing behavior.
- **Integration coverage:** unit tests cover helper branches; packaged smoke covers Electron-to-binary path.
- **Unchanged invariants:** two anchors remain required; strict calibration remains opt-in hard abort; JSON-lines progress protocol remains unchanged.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Template spacing picks neighboring image content rather than perforations | Require plausible repeated Y deltas and keep contour detector as first choice |
| Anchor-pair inference guesses the wrong interval count | Use it only after template recovery fails and bound it relative to template/perf height |
| More batches continue in fallback but produce lower-quality stabilization | Existing `calibration_status == "fallback"` warning and health-check skip communicate degraded confidence |
| Tests overfit monkeypatched helpers | Include one end-to-end synthetic fallback case through `stabilize_folder()` |

---

## Sources & References

- Origin plan: `docs/plans/2026-04-23-001-feat-stabilization-quality-pass-plan.md`
- Related code: `src/perforation_stabilizer_app.py`, `src/calibration.py`
- Related tests: `tests/test_detection.py`, `tests/test_calibration.py`
- Institutional learning: `docs/solutions/best-practices/post-refactor-review-checklist-dead-code-stale-references-2026-04-14.md`
