---
title: Fix post-refactor review findings for robust perforation tracking
type: fix
status: active
date: 2026-04-19
origin: review synthesis from commits 0f52d73..HEAD (7 reviewers)
---

# Fix post-refactor review findings for robust perforation tracking

## Overview

The robust-tracking refactor (plan `docs/plans/2026-04-18-001-refactor-robust-perforation-tracking-plan.md`, now `status: completed`) merged into local `main` with 7 reviewer personas surfacing findings. This plan resolves the P1 and P2 items before pushing to `origin`. P3 items are listed as non-goals.

## Problem Frame

The refactor replaced single-template matching with top-K candidate extraction, motion prediction (EMA), and candidate ranking. Review surfaced three classes of issues:

1. **Contract regressions**: `--smooth` flag silently ignored; `motion_rejected_count` duplicates `ambiguous_count`; ambiguous frames double-count as detection failures; log claims "uso predicción previa" but predictor position is never used.
2. **Dead code left behind by rotation removal**: `_build_rotation_template`, `_estimate_rotation`, `TestRotationEstimation` test class, stale `CLAUDE.md` architecture line, and production-dead `_template_match_perforation`.
3. **Correctness/robustness gaps**: first-frame `perf_spacing` misdetection cascades across the batch; `search_radius` floor of 120 breaks the `< perf_spacing` invariant; anchor near frame edge biases detection; hot-loop full-frame grayscale + per-frame correlation-map copy waste ~12 Mpx × 3800 work.

## Requirements Trace

- R1. `--smooth` CLI flag and `smooth_radius` parameter must either apply smoothing or be removed (no silent-ignore contract).
- R2. Dead rotation code and its test class must be removed from `src/perforation_stabilizer_app.py` and `tests/test_detection.py`.
- R3. `CLAUDE.md` architecture description must reflect the translation-only pipeline.
- R4. Ambiguous frames must not inflate `failed_detections`; `motion_rejected_count` must either track a genuinely distinct signal or be removed.
- R5. Ambiguous-frame log message must match observed behavior (either emit a real predictor fallback, or change the message).
- R6. `_detect_perf_spacing` must reject single-contour or degenerate first-frame results rather than silently returning a wrong pitch.
- R7. `search_radius` must satisfy `search_radius < perf_spacing` for all valid `perf_spacing` values.
- R8. `_template_match_perforation` must either be deleted or rewritten as a thin wrapper over `_template_match_candidates` (no duplicated sub-pixel logic).
- R9. Hot-loop BGR→GRAY conversion must operate on the matchTemplate ROI, not the full frame.
- R10. `_extract_top_k_peaks` must not allocate a full correlation-map copy per frame.
- R11. Anchor offset within template must be tracked so clipped-edge templates do not bias detection.
- R12. `_rank_candidates` dominance self-skip must use index-based comparison.
- R13. Per-frame detection loop in `stabilize_folder` must be extracted into a focused helper so the counter semantics above become reviewable.
- R14. New tests must cover: ambiguous branch end-to-end, `perf_spacing` fallback path, ROI-restricted `_template_match_candidates`, `search_radius` invariant, and anchor-at-edge clipping bias.

## Scope Boundaries

- **Non-goals (P3):** NaN-hardening of `_extract_top_k_peaks` early-exit; `_MotionPredictor` alpha surfacing to CLI; `_detect_perf_bbox` "is this plausibly a perforation?" heuristic for user misclicks; broad `except Exception` narrowing in debug block; predictor staleness reset after N ambiguous frames.
- **Non-goals (strategic):** No changes to Electron UI beyond renaming a surfaced metric (if `motion_rejected_count` goes away). No new CLI flags. No new summary fields beyond what R4 requires.
- **Out of scope:** Re-introducing rotation correction; changing the first-frame-only template strategy; switching away from NCC.

## Context & Research

### Relevant Code and Patterns

- `src/perforation_stabilizer_app.py`
  - `stabilize_folder` hot loop (lines ~780–925) — target of most changes
  - `_detect_perf_spacing` (lines ~440–495)
  - `_extract_top_k_peaks` (lines ~220–260)
  - `_template_match_candidates` (lines ~260–330)
  - `_rank_candidates` (lines ~355–440)
  - `_build_perforation_template` (lines ~130–170)
  - `_template_match_perforation` (lines ~498–594) — production-dead
  - `_build_rotation_template`, `_estimate_rotation` (lines ~597–688) — fully dead
  - `moving_average` (lines ~18–40) — dead unless R1 restores smoothing
- `tests/test_detection.py`
  - `TestRotationEstimation` class (lines ~380–442) — delete
  - Existing test structure to mirror for new coverage
- `src/stabilizer_cli.py`
  - `--smooth` flag plumbing (line 48, 104–106)
- `CLAUDE.md` line 49 — stale rotation reference
- `electron/renderer/renderer.js` — surfaces `ambiguous_frames` / `motion_rejected_frames` in the summary UI

### Institutional Learnings

- `docs/solutions/best-practices/post-refactor-review-checklist-dead-code-stale-references-2026-04-14.md` — the exact checklist this plan implements a second time (rotation removal missed these). Re-read before landing unit 1.

### External References

- None required; all fixes are local refactors or contract fixes based on existing code.

## Key Technical Decisions

- **Remove `--smooth` / `smooth_radius` rather than restore smoothing.** Dogfood on `EXPORT TEST FRAMES/` produced perf-centroid std of 7.46 px across 257 frames without smoothing — acceptable quality. Restoring a feature the new pipeline was designed around would re-introduce lag on fast motion; honest removal is cheaper than validating a restored behavior. `moving_average` helper goes with it.
- **Replace `motion_rejected_count` with a real motion-plausibility gate.** The simplest distinct signal: reject an unambiguous top candidate when its distance from `predictor.predict()` exceeds `perf_spacing × 0.5`. This gives the counter a genuine meaning and prevents the predictor from locking onto a neighbour-perf swap that happens to be unambiguous.
- **Ambiguous frames: use `predictor.predict()` as the fallback position, do not count as failure.** Aligns implementation with the log message and with the `_rank_candidates` docstring contract.
- **Delete `_template_match_perforation` outright** rather than rewriting it as a wrapper. It has no production caller; only tests exercise it. Delete function + its tests.
- **Extract `_locate_anchor_in_frame` helper** returning a `NamedTuple(pt | None, ambiguous, ranked, predicted)`. Makes counter semantics trivially auditable and unblocks new targeted tests for the ambiguous path without needing a full folder fixture.
- **Anchor-offset tracking:** `_build_perforation_template` returns `(template, (anchor_x_in_tpl, anchor_y_in_tpl))` instead of just the template. Candidates convert NCC peak → frame coordinates via the anchor offset, not `tw/2`.
- **`perf_spacing` sanity:** require ≥3 accepted centers AND `median_diff ≥ 1.5 × perf_h`; otherwise raise and force a user re-pick. Wrong-batch silent corruption is worse than a hard stop at frame 1.
- **`search_radius` formula:** `int(min(max(perf_spacing * 0.45, 80), perf_spacing - 40))`. Floor drops from 120 → 80; `min(..., perf_spacing - 40)` clamp preserves the invariant for all `perf_spacing ≥ 120`. For `perf_spacing < 120`, raise early (handled by perf_spacing sanity above).
- **BGR ROI crop before `cvtColor`:** compute crop window from `predicted ± (search_radius + tw/2 + margin)` once per frame; pass ROI origin through to `_template_match_candidates`. No full-frame grayscale in the hot loop.
- **`_extract_top_k_peaks` in-place suppression:** snapshot the 3×3 patch around each integer peak before writing `very_low`, refine sub-pixel from the snapshot. Eliminates `corr_map.copy()`.

## Open Questions

### Resolved During Planning

- **Restore or remove smoothing?** Remove (see Key Technical Decisions).
- **Re-introduce debug output for low-confidence successes?** No — keep debug emission gated on `pt is None`. Future change if needed.
- **Change Electron UI?** Rename `motion_rejected_frames` in the JSON contract stays; semantics change (now means real motion-gate rejections). UI label copy updated to "Rechazadas por movimiento (fuera de rango)" for clarity.

### Deferred to Implementation

- **Exact `perf_spacing_sq` threshold** for the new motion-plausibility gate — start with `0.5 × perf_spacing` and tune if tests show excessive false rejections.
- **Whether `_build_perforation_template` should also return `(tw, th)`** — determine during implementation; if callers already compute `tpl.shape`, skip.

## Implementation Units

- [ ] **Unit 1: Remove dead rotation code and stale docs**

**Goal:** Delete `_build_rotation_template`, `_estimate_rotation`, their tests, and the CLAUDE.md architecture line that still mentions rotation.

**Requirements:** R2, R3

**Dependencies:** None

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (remove ~90 lines: `_build_rotation_template`, `_estimate_rotation`)
- Modify: `tests/test_detection.py` (remove imports on lines 20/23; delete `TestRotationEstimation` class ~380–442)
- Modify: `CLAUDE.md` (line 49 architecture paragraph — rewrite to translation-only)

**Approach:**
- Straight deletion of rotation helpers and the test class.
- Rewrite CLAUDE.md line 49 to: "Each frame is translated (no rotation) to lock the anchor to a fixed pixel position."

**Patterns to follow:**
- `docs/solutions/best-practices/post-refactor-review-checklist-dead-code-stale-references-2026-04-14.md`

**Test scenarios:**
- Happy path: full `pytest tests/` run passes after deletion — confirms no remaining imports or references.
- Edge case: grep `_build_rotation_template` / `_estimate_rotation` across repo returns zero hits.

**Verification:**
- Test suite green; `ruff check src/` clean; no references to rotation helpers anywhere in repo except git history.

---

- [ ] **Unit 2: Remove `smooth_radius` / `--smooth` / `moving_average`**

**Goal:** Honest removal of the silently-ignored smoothing contract.

**Requirements:** R1

**Dependencies:** None

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (remove `moving_average` function; drop `smooth_radius` parameter from `stabilize_folder` signature and docstring)
- Modify: `src/stabilizer_cli.py` (remove `--smooth` argparse registration and passthrough)
- Modify: `electron/main.js` and `electron/renderer/*` (remove any `--smooth` argument construction, if present)
- Modify: `tests/test_detection.py` (remove any test that passes `smooth_radius` to `stabilize_folder`)
- Modify: `CLAUDE.md` (remove `smooth_radius` from key-parameters list)

**Approach:**
- Remove the parameter from the public signature. External callers that still pass it will raise `TypeError` — acceptable since all known callers are in-repo and updated in this unit.
- Remove `moving_average` helper; it has no other callers.

**Patterns to follow:**
- Prior cleanup in commit `2b7d4e1` for summary-field additions (mirror style).

**Test scenarios:**
- Happy path: `stabilize_folder(input_dir, output_dir, anchor=(x,y))` still runs end-to-end on a tiny fixture.
- Error path: passing `smooth_radius=9` raises `TypeError: got an unexpected keyword argument`.
- Integration: running `stabilizer_cli.py --mode batch ... --smooth 5` exits non-zero with an unrecognized-argument error.

**Verification:**
- Grep for `smooth_radius` and `moving_average` across repo returns only git history.
- Electron UI still launches a batch successfully.

---

- [ ] **Unit 3: Delete `_template_match_perforation` and its tests**

**Goal:** Remove the production-dead single-peak matcher that duplicates sub-pixel and ROI logic with `_template_match_candidates`.

**Requirements:** R8

**Dependencies:** None (but land after Unit 1/2 to minimize churn)

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (delete `_template_match_perforation`, lines ~498–594)
- Modify: `tests/test_detection.py` (delete `TestTemplateMatching` class tests that exercise `_template_match_perforation` specifically; keep any that test `_template_match_candidates` or `_subpixel_refine` directly)

**Approach:**
- Inspect each test currently importing `_template_match_perforation`; migrate meaningful assertions to `_template_match_candidates(k=1)` coverage, delete the rest.

**Test scenarios:**
- Happy path: full test suite passes after deletion.
- Edge case: grep confirms no remaining references.

**Verification:**
- Test suite green; `ruff check` clean.

---

- [ ] **Unit 4: Harden `_detect_perf_spacing` (first-frame cascade prevention)**

**Goal:** Reject degenerate first-frame perforation-spacing detection rather than silently returning a wrong pitch.

**Requirements:** R6

**Dependencies:** None

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (`_detect_perf_spacing`, lines ~440–495)
- Modify: `tests/test_detection.py` (new tests under `TestDetectPerfSpacing`)

**Approach:**
- Require ≥3 accepted contour centers before computing `np.median(np.diff(...))`.
- After median, assert `median_diff >= 1.5 * perf_h`; if not, return `None`.
- Caller (`stabilize_folder`) already raises when `perf_spacing is None` → leverage existing error path (add a clearer message: "No se detectaron al menos 3 perforaciones válidas. Re-seleccione el ancla sobre una perforación clara.").
- Remove the `first_frame.shape[0] * 0.5` heuristic fallback — it was a silent footgun.

**Test scenarios:**
- Happy path: frame with 4 bright perforations at uniform spacing returns correct pitch.
- Edge case: frame with exactly 2 perforations returns `None` (below 3-center floor).
- Edge case: frame with 3 centers where median diff < 1.5 × perf_h returns `None`.
- Error path: `stabilize_folder` raises with the new message when `_detect_perf_spacing` returns `None`.

**Verification:**
- Test suite green; dogfood run on `EXPORT TEST FRAMES/` with known-good anchor still passes.

---

- [ ] **Unit 5: Fix `search_radius` invariant**

**Goal:** Guarantee `search_radius < perf_spacing` always.

**Requirements:** R7

**Dependencies:** Unit 4 (relies on `perf_spacing` sanity floor)

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (`stabilize_folder`, line ~790)
- Modify: `tests/test_detection.py` (add `TestSearchRadiusInvariant` or inline into existing `TestStabilizeAnchor` class)

**Approach:**
- New formula: `search_radius = int(min(max(perf_spacing * 0.45, 80), perf_spacing - 40))`.
- Unit 4's perf_spacing floor (1.5 × perf_h ≈ 150+ for real scans) keeps `perf_spacing - 40 ≥ 110`, so the invariant holds for all accepted inputs.

**Test scenarios:**
- Happy path: `perf_spacing=300` → `search_radius=135`.
- Edge case: `perf_spacing=120` → `search_radius=80`, still `< perf_spacing`.
- Edge case: `perf_spacing=200` → `search_radius=90`, NOT 120 (regression lock for P2 #7).
- Property: across `perf_spacing` in `[120, 1000]`, `search_radius < perf_spacing` holds.

**Verification:**
- New property test passes; dogfood output quality unchanged.

---

- [ ] **Unit 6: Rework per-frame counter semantics and extract helper**

**Goal:** Make ambiguous / motion-rejected / failed-detection counters disjoint and meaningful; extract the per-frame detection into a helper so the logic is reviewable.

**Requirements:** R4, R5, R13

**Dependencies:** Units 4 and 5 (operates on the same hot loop)

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (extract `_locate_anchor_in_frame`; rewrite counter branches in `stabilize_folder`)
- Modify: `tests/test_detection.py` (new `TestLocateAnchorInFrame` class)
- Modify: `electron/renderer/renderer.js` (relabel the motion-rejected metric to "Rechazadas por movimiento")

**Approach:**
- Create `_FrameOutcome = collections.namedtuple("_FrameOutcome", "pt ambiguous motion_rejected ranked predicted")`.
- Extract `_locate_anchor_in_frame(gray_roi, roi_origin, template, anchor_in_tpl, predictor, search_radius, perf_spacing, min_confidence) -> _FrameOutcome`.
- Inside the helper:
  - Run candidates; rank.
  - If ambiguous: return `pt=predictor.predict()`, `ambiguous=True`, do NOT call `predictor.update()`.
  - Else if `ranked` empty: return `pt=None`.
  - Else: top candidate at distance `d` from `predicted`. If `d > perf_spacing * 0.5`: return `pt=None, motion_rejected=True` (do NOT update predictor).
  - Else: return `pt=top.xy`, call `predictor.update(pt)`.
- Hot loop now:
  - `failures += 1` ONLY when `pt is None AND NOT ambiguous AND NOT motion_rejected`.
  - `ambiguous_count` counts genuine ambiguities; predictor fallback still fills the output position.
  - `motion_rejected_count` counts real gate rejections (distinct from ambiguous).
- Ambiguous frames now contribute a predictor-based position to `points`, so `np.interp` is no longer responsible for ambiguity gaps (only for genuine failures + motion rejections).

**Patterns to follow:**
- `collections.namedtuple` usage style already present in the codebase (check and mirror); else use a small `dataclass`.

**Test scenarios:**
- Happy path: unambiguous top candidate within 0.5×perf_spacing of prediction → `pt` set, `predictor.update` called, no counters bump.
- Ambiguous: two near-equal NCC peaks at ~perf_spacing apart → `pt = predictor.predict()`, `ambiguous=True`, predictor NOT updated, `failures` not incremented.
- Motion-rejected: single top candidate at distance > 0.5×perf_spacing from prediction → `pt=None`, `motion_rejected=True`, predictor NOT updated, `failures` not incremented.
- Detection-failed: empty `ranked` → `pt=None`, both flags False, `failures` incremented.
- Integration: `stabilize_folder` summary fields `ambiguous_frames + motion_rejected_frames + failed_detections ≤ total_frames` AND these three sets are disjoint.
- Regression lock: a frame that used to be counted as both ambiguous AND failed now counts only as ambiguous.

**Verification:**
- Test suite green.
- Electron UI displays the three counters with non-overlapping semantics on a real dogfood batch.
- Log message for ambiguous frames no longer lies — "uso predicción previa" now matches the code path.

---

- [ ] **Unit 7: Anchor-offset tracking in template**

**Goal:** Preserve the `user click → fixed pixel` contract when the user clicks near a frame edge.

**Requirements:** R11

**Dependencies:** Unit 6 (`_locate_anchor_in_frame` signature change)

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (`_build_perforation_template` returns `(template, anchor_in_tpl)`; `_template_match_candidates` accepts `anchor_in_tpl` and converts peak → frame coords using it)
- Modify: `tests/test_detection.py` (new `TestAnchorOffsetEdge` tests)

**Approach:**
- `_build_perforation_template` tracks `anchor_x_in_tpl = anchor_x - x0`, `anchor_y_in_tpl = anchor_y - y0` (where `(x0, y0)` is the template origin after edge clipping).
- Candidate peak-to-frame conversion becomes `cx = roi_x0 + sx + anchor_x_in_tpl` instead of `roi_x0 + sx + tw/2`.
- No change to callers beyond wiring the offset tuple through.

**Test scenarios:**
- Happy path: user click at frame center → `anchor_in_tpl == (tw/2, th/2)`; detection unchanged vs prior behavior.
- Edge case: user click at `(50, 50)` with `patch_radius=60` → template clipped, `anchor_in_tpl == (50, 50)`; detected position on a second identical frame equals `(50, 50)`.
- Regression: stabilized output for an edge-clicked anchor places that exact pixel at the anchor coords across frames (tolerance ±1 px).

**Verification:**
- Edge-anchor integration test passes; dogfood run with a centered anchor unchanged.

---

- [ ] **Unit 8: BGR ROI crop before grayscale conversion**

**Goal:** Reduce hot-loop cvtColor from 12 Mpx to <1% of current work.

**Requirements:** R9

**Dependencies:** Unit 6 (clean helper boundary to pass ROI through)

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (`stabilize_folder` hot loop; `_template_match_candidates` already accepts `search_center`/`search_radius` — augment to accept a pre-cropped ROI with an explicit origin)
- Modify: `tests/test_detection.py` (add perf-bench style test comparing full-frame vs ROI path output equivalence)

**Approach:**
- Compute `crop = (cx - r, cy - r, cx + r, cy + r)` where `r = search_radius + template_half + margin` and `(cx, cy) = predictor.predict()`.
- Clip crop to frame bounds; `gray_roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)`.
- Pass `gray_roi` + `roi_origin=(x0, y0)` into `_template_match_candidates`; internal matchTemplate runs on that ROI directly; peak coords translated by `roi_origin` before returning.

**Test scenarios:**
- Happy path: detected positions on a known fixture match the previous full-frame path to within sub-pixel tolerance (`< 0.01 px`).
- Edge case: crop that touches frame boundary is clipped correctly; no out-of-bounds read.
- Edge case: crop that would extend beyond the frame on one side produces a correctly-offset ROI (origin reflects the actual crop, not the requested crop).

**Verification:**
- Equivalence test passes.
- Dogfood run on `EXPORT TEST FRAMES/` completes with the same `failed_detections` count as before; wall-clock improvement observable (expected ≥3× on 4K batches, but time-bound not hard-enforced).

---

- [ ] **Unit 9: In-place suppression in `_extract_top_k_peaks`**

**Goal:** Eliminate per-frame `corr_map.copy()` allocation.

**Requirements:** R10

**Dependencies:** None (isolated function change)

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (`_extract_top_k_peaks`)
- Modify: `tests/test_detection.py` (add test proving input `corr_map` is NOT mutated by the function — the public contract stays "non-mutating to caller")

**Approach:**
- Before overwriting the suppression window with `very_low`, snapshot the 3×3 patch around `(mx, my)` into a local `np.ndarray`.
- Pass the snapshot into `_subpixel_refine`.
- Caller passes a writable `corr_map`; function restores original values for returned peaks' 3×3 patches before returning (or the function accepts that `corr_map` is consumed — simpler: accept consumption, rename parameter to signal it, and have `_template_match_candidates` feed it the `matchTemplate` output directly without a separate copy).
- Decision at impl time: cheaper option is "consume the corr_map" — `_template_match_candidates` never re-uses it after peak extraction.

**Test scenarios:**
- Happy path: three planted peaks still extracted correctly in descending score order (regression of existing test).
- Edge case: `k=1` returns exactly one peak and does not touch the rest of the surface.
- Edge case: all-constant correlation map → returns at most 1 peak then bails via existing `isfinite` guard.

**Verification:**
- Existing top-K tests pass unchanged.
- Memory profile on a 3800-frame run shows no per-frame correlation-map allocation in `_extract_top_k_peaks`.

---

- [ ] **Unit 10: Index-based self-skip in `_rank_candidates`**

**Goal:** Replace fragile float-equality self-identity with index-based comparison.

**Requirements:** R12

**Dependencies:** None

**Files:**
- Modify: `src/perforation_stabilizer_app.py` (`_rank_candidates`, lines ~403–417)
- Modify: `tests/test_detection.py` (add test with two candidates at identical float coords — historically could trigger the wrong self-skip)

**Approach:**
- Change outer/inner loops to `enumerate` and skip `if j == i`.
- Replace `sqrt(dx*dx + dy*dy)` comparison against `perf_spacing` with squared-distance comparison against `perf_spacing_sq = perf_spacing ** 2`.

**Test scenarios:**
- Happy path: existing `test_ambiguity_flagged_on_twin_peaks` still passes.
- Edge case: two candidates with exactly equal coords but different NCC → both see each other as competing, neither is skipped as self-for-the-other.
- Regression: ranking output order unchanged for typical inputs.

**Verification:**
- Tests green.

---

- [ ] **Unit 11: Expand test coverage for review-identified gaps**

**Goal:** Lock in the fixes with tests the review flagged as missing.

**Requirements:** R14

**Dependencies:** Units 4–10 (tests reference the new behavior)

**Files:**
- Modify: `tests/test_detection.py`

**Approach:**
- Add tests covering:
  - `_template_match_candidates` ROI-restricted path (`search_center`/`search_radius` passed explicitly).
  - `_rank_candidates` with `perf_spacing=None` and `perf_spacing<=0` (NCC-only fallback).
  - `_rank_candidates` dominance where one competing peak sits inside `perf_spacing` (down-weighting verified).
  - `_MotionPredictor` bootstrap: `predict()` before any `update()` returns `initial_pos`.
  - `_MotionPredictor` alpha=0.0 and alpha=1.0 boundaries.
  - End-to-end `stabilize_folder` with a fabricated sequence that exercises the ambiguous branch (mock `_rank_candidates` or build a synthetic twin-peak fixture).
  - Non-feature test: assert `stabilize_folder` warp matrix contains only translation (no rotation) — regression lock for P1 dead-rotation-code reintroduction.

**Test scenarios:**
- Each bullet above is itself a scenario.

**Verification:**
- `pytest tests/` passes with ≥10 net new tests; `ruff format --check` clean.

---

- [ ] **Unit 12: Dogfood + commit + push**

**Goal:** Verify on real data, then publish.

**Requirements:** All

**Dependencies:** Units 1–11

**Files:**
- None (verification-only)

**Approach:**
- Run `uv run pytest tests/` — must be green.
- Run `uv run ruff check src/ && uv run ruff format --check src/` — must be clean.
- Run the CLI batch on `EXPORT TEST FRAMES/` with anchor `(564, 600)` (the known-good bright-perforation anchor from the prior dogfood) and confirm:
  - `failed_detections == 0`
  - `ambiguous_frames + motion_rejected_frames ≤ 3`
  - Perf-centroid std across output frames ≤ 10 px
- Rebuild DMGs per `feedback_build_dmgs.md` memory; ad-hoc codesign per `feedback_codesign.md`.
- Commit as a single `fix:` commit or grouped `fix:` commits per unit (author's judgment at commit time).
- `git push origin main` (user has not pushed yet).

**Test scenarios:**
- Happy path: dogfood summary meets acceptance thresholds.
- Error path: any threshold miss → investigate before pushing.

**Verification:**
- Push succeeds; CI green on `main`.

## System-Wide Impact

- **Interaction graph:** Hot loop in `stabilize_folder` is the only runtime path; Electron UI consumes the summary JSON; no background jobs.
- **Error propagation:** Unit 4 tightens first-frame failure → `stabilize_folder` raises → CLI emits `{"type":"error"}` → Electron surfaces to user. Path already tested.
- **State lifecycle risks:** Unit 6 fixes a subtle one — predictor state contamination from ambiguous frames is already handled correctly, but motion-rejected frames will now also skip `predictor.update()`. Verify the predictor cannot drift to an off-frame prediction over a long motion-rejected streak (see testing gap below).
- **API surface parity:** CLI JSON contract gains no new keys; `motion_rejected_frames` stays but means something real now. Electron UI relabel lands with Unit 6.
- **Integration coverage:** Unit 11 explicitly adds the end-to-end ambiguous-branch integration test the review flagged.
- **Unchanged invariants:** Translation-only warp; first-frame-only template; `--anchor-x`/`--anchor-y` CLI contract; preview mode; DMG build pipeline.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Unit 4's stricter perf_spacing rejection breaks batches that currently silently succeed with wrong spacing | Dogfood on `EXPORT TEST FRAMES/` before merging; if a real batch fails, the error message tells the user exactly what to do (re-pick anchor) |
| Unit 8 ROI crop math introduces an off-by-one that biases every frame | Equivalence test in Unit 8 compares full-frame vs ROI paths to sub-pixel precision on a real fixture |
| Unit 6 counter semantics change confuses users comparing reports before/after | One-time UX cost; the new semantics are correct. Release notes in commit body |
| Removing `--smooth` breaks an external script or user muscle memory | No known external consumer; Electron UI owned by same user. CLI will error loudly rather than silently ignore |
| Unit 9 in-place `corr_map` mutation breaks `_template_match_candidates` if it ever reuses the map | Explicitly document consumption in docstring; grep confirms single use site |

## Documentation / Operational Notes

- `CLAUDE.md` architecture + parameters sections updated in Units 1 and 2.
- No user-facing release notes required (local-only pre-push).
- After Unit 12 push, consider tagging a release if client testing timeline warrants (per project memory `feedback_codesign.md`, ad-hoc signing is mandatory before release).

## Sources & References

- **Review synthesis** (this conversation, 7 reviewer personas: correctness, maintainability, testing, performance, adversarial, kieran-python, project-standards)
- **Prior plan:** `docs/plans/2026-04-18-001-refactor-robust-perforation-tracking-plan.md` (completed — defines the code being fixed here)
- **Institutional learning:** `docs/solutions/best-practices/post-refactor-review-checklist-dead-code-stale-references-2026-04-14.md`
- **Project memory:** `feedback_build_dmgs.md`, `feedback_codesign.md`
- **Diff base:** `0f52d73083c87cbb8e3b7517e98f0e8f72d95932..HEAD`
