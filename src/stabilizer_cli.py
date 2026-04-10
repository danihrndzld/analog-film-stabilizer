#!/usr/bin/env python3
"""
CLI backend for the Electron UI.
Streams JSON-lines to stdout so Electron can consume progress and logs.

Modes:
  batch (default)
    --input DIR --output DIR [--roi F] [--threshold N] [--smooth N]
    [--quality N] [--film-format super8|8mm|super16] [--debug-frames DIR]
    [--manual-anchor-x N --manual-anchor-y N]

  preview
    --frame-path FILE --preview-out FILE [--roi F] [--threshold N]
    [--film-format super8|8mm|super16]

Output lines (one per line, each valid JSON):
  Batch mode:
    {"type": "progress", "value": 0.0..1.0}
    {"type": "log",      "msg":  "..."}
    {"type": "done",     "summary": {...}}
    {"type": "error",    "msg":  "..."}

  Preview mode (single line):
    {"type": "preview", "detected": true|false,
     "cx": N|null, "cy": N|null, "previewPath": "..."|null}
"""

import sys
import json
import argparse
import math
import os

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perforation_stabilizer_app import (
    _annotate_roi_preview,
    detect_perforation,
    stabilize_folder,
)


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def run_preview(args):
    """Run single-frame detection and save an annotated preview JPEG."""
    frame = cv2.imread(args.frame_path)
    if frame is None:
        emit({"type": "preview", "detected": False,
              "cx": None, "cy": None, "previewPath": None})
        return

    h, w = frame.shape[:2]
    roi_w = max(50, int(w * args.roi))
    roi_bgr = frame[:, :roi_w]
    frame_name = os.path.basename(args.frame_path)

    anchor = detect_perforation(frame, roi_ratio=args.roi,
                                film_format=args.film_format)

    # Annotate the ROI crop; rejections list is empty when detection succeeds,
    # and also empty on failure (rejection boxes require internal binary data
    # not exposed through detect_perforation's public interface — the
    # "NO DETECTADO" banner is still shown without them).
    annotated = _annotate_roi_preview(roi_bgr, anchor,
                                      rejections=[], frame_name=frame_name)
    cv2.imwrite(args.preview_out, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

    emit({
        "type":        "preview",
        "detected":    anchor is not None,
        "cx":          float(anchor[0]) if anchor is not None and math.isfinite(anchor[0]) else None,
        "cy":          float(anchor[1]) if anchor is not None and math.isfinite(anchor[1]) else None,
        "previewPath": args.preview_out,
    })


def run_batch(args):
    """Run the full stabilisation batch."""
    manual_anchor = None
    if args.manual_anchor_x is not None and args.manual_anchor_y is not None:
        manual_anchor = (args.manual_anchor_x, args.manual_anchor_y)

    try:
        summary = stabilize_folder(
            input_dir=args.input,
            output_dir=args.output,
            progress_cb=lambda v: emit({"type": "progress", "value": v}),
            log_cb=lambda m: emit({"type": "log", "msg": m}),
            roi_ratio=args.roi,
            threshold=args.threshold,
            smooth_radius=args.smooth,
            jpeg_quality=args.quality,
            film_format=args.film_format,
            debug_dir=args.debug_frames,
            manual_anchor=manual_anchor,
            border_mode=args.border_mode,
        )
        emit({"type": "done", "summary": summary})
    except Exception as exc:
        emit({"type": "error", "msg": str(exc)})
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Perforation stabilizer CLI")

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument("--mode", choices=["batch", "preview"], default="batch",
                        help="Operation mode: batch (default) or preview")

    # ── Shared params (both modes) ────────────────────────────────────────────
    parser.add_argument("--roi",         type=float, default=0.22,
                        help="ROI fraction (default 0.22)")
    parser.add_argument("--threshold",   type=int,   default=210,
                        help="Brightness threshold (default 210)")
    parser.add_argument("--film-format", choices=["super8", "8mm", "super16"],
                        default="super8",
                        help="Film format: super8 (default), 8mm, super16")

    # ── Batch-mode params ─────────────────────────────────────────────────────
    parser.add_argument("--input",       default=None, help="Input folder with frames")
    parser.add_argument("--output",      default=None, help="Output folder")
    parser.add_argument("--smooth",      type=int,   default=9,
                        help="Moving-average radius (default 9)")
    parser.add_argument("--quality",     type=int,   default=95,
                        help="JPEG quality 1-100, 0=PNG (default 95)")
    parser.add_argument("--debug-frames", default=None, metavar="DIR",
                        help="Optional folder for failed-frame debug JPEGs")
    parser.add_argument("--manual-anchor-x", type=float, default=None,
                        help="Override target anchor X (skips median computation)")
    parser.add_argument("--manual-anchor-y", type=float, default=None,
                        help="Override target anchor Y (skips median computation)")
    parser.add_argument("--border-mode", choices=["replicate", "constant", "reflect"],
                        default="replicate",
                        help="Border fill mode for warpAffine (default: replicate)")

    # ── Preview-mode params ───────────────────────────────────────────────────
    parser.add_argument("--frame-path",  default=None,
                        help="Path to single frame for preview mode")
    parser.add_argument("--preview-out", default=None,
                        help="Output path for annotated preview JPEG")

    args = parser.parse_args()

    if args.mode == "preview":
        if not args.frame_path or not args.preview_out:
            emit({"type": "error",
                  "msg": "--frame-path and --preview-out are required in preview mode"})
            sys.exit(1)
        run_preview(args)
    else:
        if not args.input or not args.output:
            emit({"type": "error",
                  "msg": "--input and --output are required in batch mode"})
            sys.exit(1)
        run_batch(args)


if __name__ == "__main__":
    main()
