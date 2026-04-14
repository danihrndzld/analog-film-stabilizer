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

Pure processing module (no UI). Two-stage pipeline:

1. **Template building** — `_build_perforation_template()` extracts a grayscale patch (default 60px radius) around the user-selected anchor point from the first frame. This patch becomes the reference template for alignment.

2. **Stabilization pass** — `stabilize_folder()` requires a user-provided `anchor` (x, y) tuple. It builds the template from the first frame, then uses `_template_match_perforation()` (normalized cross-correlation with sub-pixel precision via parabolic fitting) to locate the anchor in every frame. Failed frames are interpolated from neighbors. Each frame is translated (and optionally rotation-corrected via `_estimate_rotation()` using ECC) to lock the anchor to a fixed position. The user's anchor IS the target position.

Key parameters:
- **`anchor`**: (x, y) tuple — user-selected reference point (required)
- **`smooth_radius`**: moving average window half-size — default `9`
- **`jpeg_quality`**: JPEG 1–100 or `0` for PNG lossless
- **`border_mode`**: `'replicate'`, `'constant'`, or `'reflect'`

Debug frames: pass `debug_dir` to `stabilize_folder()` to get `_debug.jpg` patches for failed template matches.

### Electron UI (`electron/`)

Film-frame drag-drop interface. User selects a folder, clicks a reference point on the preview of the first frame, then runs stabilization. Spawns `src/stabilizer_cli.py` as a Python subprocess and communicates over stdout (JSON progress lines). Files: `main.js` (Electron main process), `preload.js` (contextBridge IPC), `renderer/` (HTML/CSS/JS frontend). Version is read from `package.json` at runtime via `window.api.version`.

### CLI backend (`src/stabilizer_cli.py`)

Used by the Electron UI. Two modes, both emit JSON-lines to stdout:

**Batch mode** (default):
```
--input DIR --output DIR --anchor-x N --anchor-y N [--smooth N] [--quality N] [--debug-frames DIR] [--border-mode STR]
```

**Preview mode** (saves first frame as preview JPEG):
```
--mode preview --frame-path FILE --preview-out FILE
```

### CI / Build
`.github/workflows/build.yml` builds macOS DMGs (arm64 + x64) on every push to `main` and on version tags (`v*`) using electron-builder. CI also runs `ruff check` and `ruff format --check` on `src/`. Python is bundled into standalone binaries via PyInstaller (separate arm64/x64 builds). Tagged builds create a GitHub Release. Version is synced from the git tag to `package.json` automatically by CI.

Output is written to a sibling folder suffixed `_ESTABILIZADO`, plus a `stabilization_report.txt` with frame count, failed detections, anchor coords, output dimensions, and output format.

## Documented Solutions

`docs/solutions/` — documented solutions to past problems (bugs, best practices, workflow patterns), organized by category with YAML frontmatter (`module`, `tags`, `problem_type`). Relevant when implementing or debugging in documented areas.

## Media

Sample frames live in `media/` and are excluded from git (see `.gitignore`).
