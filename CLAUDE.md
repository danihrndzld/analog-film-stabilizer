# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install Python deps (uses uv, not pip)
uv sync                        # production deps
uv sync --group dev            # + ruff linter

# Run the Electron app (requires Python deps installed)
cd electron && npm start

# Tests
uv run pytest tests/           # all tests
uv run pytest tests/test_detection.py::TestTemplateMatching  # single class
uv run pytest tests/test_detection.py::TestAnchorWorkflow::test_stabilize_with_anchor_produces_output -v  # single test

# Lint
uv run ruff check src/         # lint errors
uv run ruff format --check src/ # format check
uv run ruff check --fix src/   # auto-fix
uv run ruff format src/        # auto-format

# Local build (DMGs → dist/)
./scripts/build-local.sh       # or: cd electron && npm run build
```

## Client Requirements (Diego Fabriccio)

The client scans analog film (3500–3800 frames per folder) and currently stabilizes the sequence manually in DaVinci Resolve by tracking a reference point on the film perforation. He wants this fully automated:

- **Drag-and-drop a folder** of frames — no manual per-frame work
- **Click a reference point** on the preview of the first frame
- The app locks that reference to a **fixed pixel position** across all frames via template matching
- Output is a stabilized image sequence ready to be assembled into video

**Scale consideration:** batches are 3500–3800 JPEGs. Performance and memory efficiency matter.

## Architecture

### Python backend (`src/perforation_stabilizer_app.py`)

Pure processing module (no UI). Main pipeline:

1. **Calibration/template building** — `run_calibration()` samples the batch to build calibrated templates and perforation spacing. If calibration cannot stabilize, normal mode falls back to first-frame bootstrap with a warning; `strict_calibration=True` aborts.

2. **Two-anchor stabilization pass** — `stabilize_folder()` requires `anchor1` and `anchor2` `(x, y)` tuples. Each anchor is tracked independently with `_template_match_candidates()` (normalized cross-correlation with sub-pixel precision via parabolic fitting, top-K peaks), motion-predictor ranking (`_rank_candidates`), and pair-consensus gates. Raw transforms use both anchors when available, fall back to translation-only from one surviving anchor, then pass through splice-aware smoothing before warp. The warp remains translation-only.

3. **R13 health check** — after Pass 1 and splice detection, the pipeline computes rejection pressure from motion rejections, consensus rejections, and NaN-filled frames. Normal mode logs/report-writes a warning above `reject_ceiling`; `strict_health_check=True` raises `HealthCheckError`.

Key parameters:
- **`anchor1`, `anchor2`**: (x, y) tuples — user-selected reference points (required)
- **`jpeg_quality`**: JPEG 1–100 or `0` for PNG lossless
- **`border_mode`**: `'replicate'`, `'constant'`, or `'reflect'`
- **`strict_calibration`**: hard-abort instead of calibration fallback
- **`reject_ceiling`**: R13 rejection-rate warning threshold, default `0.20`
- **`strict_health_check`**: hard-abort when R13 exceeds `reject_ceiling`

Debug frames: pass `debug_dir` to `stabilize_folder()` to get `_debug.jpg` patches for failed template matches.

### Electron UI (`electron/`)

Film-frame drag-drop interface. User selects a folder, clicks a reference point on the preview of the first frame, then runs stabilization. Spawns `src/stabilizer_cli.py` as a Python subprocess and communicates over stdout (JSON progress lines). Files: `main.js` (Electron main process), `preload.js` (contextBridge IPC), `renderer/` (HTML/CSS/JS frontend). Version is read from `package.json` at runtime via `window.api.version`.

### CLI backend (`src/stabilizer_cli.py`)

Used by the Electron UI. Two modes, both emit JSON-lines to stdout:

**Batch mode** (default):
```
--input DIR --output DIR --anchor1-x N --anchor1-y N --anchor2-x N --anchor2-y N
[--quality N] [--debug-frames DIR] [--border-mode STR]
[--strict-calibration] [--reject-ceiling 0.20] [--strict-health-check]
```

**Preview mode** (saves first frame as preview JPEG):
```
--mode preview --frame-path FILE --preview-out FILE
```

### CI / Build
`.github/workflows/build.yml` builds macOS DMGs (arm64 + x64) on every push to `main` and on version tags (`v*`) using electron-builder. CI also runs `ruff check` and `ruff format --check` on `src/`. Python is bundled into standalone binaries via PyInstaller (separate arm64/x64 builds). Tagged builds create a GitHub Release. Version is synced from the git tag to `package.json` automatically by CI.

Output is written to a sibling folder suffixed `_ESTABILIZADO`, plus a `stabilization_report.txt` with frame count, rejection counters, calibration/health status, anchor coords, output dimensions, and output format.

## Documented Solutions

`docs/solutions/` — documented solutions to past problems (bugs, best practices, workflow patterns), organized by category with YAML frontmatter (`module`, `tags`, `problem_type`). Relevant when implementing or debugging in documented areas.

## Media

Sample frames live in `media/` and are excluded from git (see `.gitignore`).
