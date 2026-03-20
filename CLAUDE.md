# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
cd electron && npm start
```

Requires Python 3 with `opencv-python` and `numpy` installed:
```bash
pip3 install opencv-python numpy
```

## Client Requirements (Diego Fabriccio)

The client scans analog film (3500–3800 frames per folder) and currently stabilizes the sequence manually in DaVinci Resolve by tracking a reference point on the film perforation. He wants this fully automated:

- **Drag-and-drop a folder** of frames — no manual per-frame work
- **Select film format** once (Super 8, 8mm Regular, Super 16)
- The app locks the perforation reference to a **fixed pixel position** across all frames
- Output is a stabilized image sequence ready to be assembled into video

**Scale consideration:** batches are 3500–3800 JPEGs. Performance and memory efficiency matter.

## Architecture

### Python backend (`src/perforation_stabilizer_app.py`)

Pure processing module (no UI). Two processing passes:

1. **Detection pass** — `detect_perforation()` crops a ROI (default 22% of frame width) on the left or right side depending on film format, applies Gaussian blur + binary threshold, morphological open/close, then scores contours by area + fill ratio. Filters: area ≥ 5000 px, aspect ratio 0.40–1.20, fill ≥ 0.75, centroid within 70% of ROI width. For 8mm Regular, finds the top-2 contours and returns their midpoint as the anchor. Falls back to adaptive threshold for overexposed frames.

2. **Stabilization pass** — `stabilize_folder()` uses the **median** of raw per-frame positions as the fixed anchor, fills missed frames by linear interpolation, then translates each frame with `cv2.warpAffine`.

Key parameters:
- **`film_format`**: `'super8'` (1 perf, left ROI), `'8mm'` (2 perfs, right ROI, midpoint anchor), `'super16'` (1 perf, right ROI)
- **`roi_ratio`**: fraction of frame width to search — default `0.22`
- **`threshold`**: brightness cutoff for binarization — default `210`
- **`smooth_radius`**: moving average window half-size — default `9`
- **`jpeg_quality`**: JPEG 1–100 or `0` for PNG lossless

### Electron UI (`electron/`)

Film-frame drag-drop interface. Spawns `src/stabilizer_cli.py` as a Python subprocess and communicates over stdout (JSON progress lines). Files: `main.js` (Electron main process), `preload.js` (contextBridge IPC), `renderer/` (HTML/CSS/JS frontend). Version is read from `package.json` at runtime via `window.api.version`.

### CLI backend (`src/stabilizer_cli.py`)

Used by the Electron UI. Accepts folder path and parameters as CLI args; emits JSON progress to stdout.

```
--input DIR --output DIR [--roi F] [--threshold N] [--smooth N] [--quality N] [--film-format super8|8mm|super16]
```

### CI / Build
`.github/workflows/build.yml` builds macOS DMGs (arm64 + x64) on every push to `main` and on version tags (`v*`) using electron-builder. Artifacts are uploaded for 30 days; tagged builds create a GitHub Release. Version is synced from the git tag to `package.json` automatically by CI.

Output is written to a sibling folder suffixed `_ESTABILIZADO`, plus a `stabilization_report.txt` with frame count, failed detections, anchor coords, output dimensions, and film format.

## Media

Sample frames live in `media/` and are excluded from git (see `.gitignore`).
