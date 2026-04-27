---
title: Post-Refactor Review Checklist — Dead Code, Stale References, and Silent Failures
date: "2026-04-14"
category: best-practices
module: full-stack (python-backend, electron-ui, cli, tests)
problem_type: best_practice
component: development_workflow
severity: medium
applies_when:
  - "After removing a major feature or subsystem that touched multiple layers"
  - "When replacing one detection/alignment strategy with another across backend, CLI, and UI"
  - "When a refactor spans Python backend, Electron renderer, HTML, CSS, and test files"
  - "Post-refactor review to catch orphaned code, stale selectors, and silent failures"
tags:
  - refactor
  - code-review
  - dead-code
  - template-matching
  - opencv
  - electron
  - stale-references
  - magic-numbers
---

# Post-Refactor Review Checklist — Dead Code, Stale References, and Silent Failures

## Context

A major refactor of an analog film stabilizer app replaced ~600 lines of automatic contour-based perforation detection (supporting Super 8, 8mm Regular, Super 16 formats) with a simpler user-selected anchor point tracked via OpenCV template matching (NCC with sub-pixel parabolic refinement) and ECC rotation estimation. The pipeline was simplified from three stages (bootstrap averaging from 15 frames, contour detection, stabilization) to two stages (template from first frame at user click, template matching across all frames).

A structured code review of the refactored codebase found 8 safe-to-fix issues and 3 pre-existing advisory issues that the refactor surfaced but did not introduce.

## Guidance

### 1. Grep for orphaned references after deleting a subsystem

After deleting a major subsystem, run a structured review specifically looking for orphaned references: CSS classes, JS functions, Python variables, and parameters that only existed to support the removed code. Automated linting catches unused imports but not stale DOM class references or parameters accepted but never used.

### 2. Deduplicate per-item operations in hot loops

When two functions in a hot loop both convert the same input (e.g., BGR to grayscale), refactor the conversion to happen once before the loop and pass the converted form to both. For 3500+ frame batches, this eliminates thousands of redundant `cv2.cvtColor` calls.

### 3. Replace magic numbers that duplicate defaults

Replace hardcoded literals that duplicate a default parameter value with a derivation from the actual data. A `pr = 60` that duplicates `patch_radius=60` should use `template.shape[0] // 2` instead, tying the debug crop to the real template size.

### 4. Guard numeric values crossing process boundaries

Guard numeric values crossing process boundaries (Electron to Python subprocess) with explicit finiteness checks. `parseInt` on an empty or non-numeric string returns `NaN`, which propagates silently into subprocess command line args and causes a crash downstream.

### 5. Never silently swallow exceptions

Bare `except: pass` blocks hide real failures. At minimum, log at debug level so failures surface when investigating issues without disrupting the normal path.

### 6. Audit every parameter the public API accepts

When simplifying a pipeline, explicitly audit every parameter the public API accepts. A parameter that was wired in the old architecture (`smooth_radius` feeding `moving_average()`) can silently become dead weight after the refactor, giving users a knob that does nothing.

## Why This Matters

Large deletions create a false sense of cleanliness. The diff looks smaller and simpler, so reviewers assume the remaining code is tight. In practice, refactors that remove a subsystem reliably leave behind orphaned references, redundant operations that were masked by the old code's complexity, and parameters that lost their downstream consumer.

In a batch-processing app handling 3500+ frames, a single redundant per-frame operation (like a duplicate color conversion) multiplies into minutes of wasted wall-clock time. Unguarded cross-process value passing (Electron to Python via CLI args) creates failure modes that only appear with specific user input combinations and produce no useful error message.

## When to Apply

- After removing a major feature or subsystem (especially one that spanned multiple layers: backend, frontend, CLI)
- When refactoring a multi-stage pipeline down to fewer stages
- When code passes numeric parameters across process boundaries (IPC, subprocess args, HTTP params)
- When a hot loop calls multiple functions that each independently prepare the same input data
- During post-refactor review of any codebase processing large batches where per-item overhead compounds

## Examples

**Redundant conversion (performance):**
`_template_match_perforation` and `_estimate_rotation` each called `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` on the same frame inside the stabilization loop. Fix: convert once before calling either function, pass grayscale to both.

**Orphaned frontend reference (dead code):**
`classList.remove('is-outside-roi')` referenced a CSS class deleted with the ROI subsystem. No error thrown (DOM classList operations are silent), but dead code obscured what the UI actually does.

**Dead utility function (dead code):**
`escapeHtml()` was left behind after an innerHTML-to-DOM-API migration. Never called, but added 8 lines of noise to the renderer.

**Unguarded cross-process numeric (crash path):**
`parseInt(qualityInput.value)` could yield `NaN` when the input was empty; passed as a CLI arg, Python's `int()` raised `ValueError`. Fix: `Number.isFinite()` guard, preserving `0` as a valid value (PNG lossless mode).

**Silent parameter (correctness):**
`smooth_radius` accepted by `stabilize_folder()` and passed through from the CLI, but `moving_average()` was never called after the contour detection removal. The parameter became a no-op with no warning.

## Related

- `docs/plans/2026-04-14-001-refactor-remove-film-formats-user-anchor-plan.md` — the plan that drove this refactor
- `docs/plans/2026-04-08-002-fix-8mm-single-perf-anchor-and-preview-plan.md` — created the click-to-anchor UI this refactor made the sole workflow
- `docs/plans/2026-04-11-001-feat-corner-anchor-and-expanded-preview-plan.md` — intermediate anchor iteration, superseded by this refactor
