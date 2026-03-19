# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

Double-click `src/Perforation_Stabilizer.command` on macOS, or run directly:

```bash
python3 src/perforation_stabilizer_app.py
```

Dependencies are auto-installed on first run (`opencv-python`, `numpy`, `tkinterdnd2`).

If the launcher is blocked by macOS:
```bash
chmod +x src/Perforation_Stabilizer.command
```

## Client Requirements (Diego Fabriccio)

The client scans analog film (3500–3800 frames per folder) and currently stabilizes the sequence manually in DaVinci Resolve by tracking a reference point on the film perforation. He wants this fully automated:

- **Drag-and-drop a folder** of frames — no manual per-frame work
- **Mark the search area** once (or let the app auto-detect it) — the user selects where the perforation is, not every frame
- The app locks that reference point to a **fixed pixel position** across all frames
- Output is a stabilized image sequence ready to be assembled into video

The key workflow analogy: "I select the lower-right corner of the perforation in DaVinci and it finds that in all photos and keeps that point fixed." The app should replicate this automatically.

**Scale consideration:** batches are 3500–3800 JPEGs. Performance and memory efficiency matter.

## Architecture

Single-file Tkinter GUI app (`src/perforation_stabilizer_app.py`) with two processing passes:

1. **Detection pass** — `detect_perforation()` crops a left-side ROI (default 22% of frame width), applies Gaussian blur + binary threshold, morphological open/close, then scores contours by area + fill ratio. Filters: area ≥ 5000 px, aspect ratio 0.40–1.20, fill ≥ 0.75, centroid within 70% of ROI width. Returns centroid of the best candidate.
2. **Stabilization pass** — `stabilize_folder()` applies `moving_average()` (uniform kernel, NaN-filled via linear interpolation for missed frames) to smooth positions, uses the **median** of smoothed points as the fixed anchor, then translates each frame with `cv2.warpAffine`. Before the second-pass loop, all shifts are pre-computed and used to derive a symmetric crop (`crop_left/right/top/bottom`) that removes every black border across the whole batch. Output frames are smaller than input by those amounts and have no black borders.

Key parameters exposed in the UI:
- **ROI izq.** (`roi_ratio`): fraction of frame width to search — default `0.22`
- **Threshold**: brightness cutoff for binarization — default `210` (lower if missing detections, raise if false positives)
- **Suavizado** (`smooth_radius`): moving average window half-size — default `9`

### UI classes
- `DragDropApp` — used when `tkinterdnd2` loads successfully; adds a drag-and-drop target zone.
- `PickerApp` — fallback when DnD is unavailable; uses file-picker buttons only.
- Both inherit `AppBase` which owns `run_process()`, `log()`, `set_progress()`, `_finish()`, `_error()`.

Processing runs in a daemon thread; all UI updates go through `root.after()`.

### CI / Build
`.github/workflows/build.yml` builds a macOS `.app` bundle on every push to `main` and on version tags (`v*`) using PyInstaller (`--windowed --onedir`). Artifacts are uploaded for 30 days; tagged builds create a GitHub Release with the zip attached.

Output is written to a sibling folder suffixed `_ESTABILIZADO`, plus a `stabilization_report.txt` with frame count, failed detections, anchor coords, output dimensions, and applied crop values (left/right/top/bottom px).

## Media

Sample frames live in `media/` and are excluded from git (see `.gitignore`).
