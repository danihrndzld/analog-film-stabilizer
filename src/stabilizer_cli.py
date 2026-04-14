#!/usr/bin/env python3
"""
CLI backend for the Electron UI.
Streams JSON-lines to stdout so Electron can consume progress and logs.

Modes:
  batch (default)
    --input DIR --output DIR --anchor-x N --anchor-y N
    [--smooth N] [--quality N] [--debug-frames DIR] [--border-mode STR]

  preview
    --frame-path FILE --preview-out FILE

Output lines (one per line, each valid JSON):
  Batch mode:
    {"type": "progress", "value": 0.0..1.0}
    {"type": "log",      "msg":  "..."}
    {"type": "done",     "summary": {...}}
    {"type": "error",    "msg":  "..."}

  Preview mode (single line):
    {"type": "preview", "previewPath": "..."|null}
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perforation_stabilizer_app import (
    stabilize_folder,
)


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def run_preview(args):
    """Save a preview JPEG of the first frame for the UI."""
    frame = cv2.imread(args.frame_path)
    if frame is None:
        emit({"type": "preview", "previewPath": None})
        return

    cv2.imwrite(args.preview_out, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    emit({"type": "preview", "previewPath": args.preview_out})


def run_batch(args):
    """Run the full stabilisation batch."""
    anchor = (args.anchor_x, args.anchor_y)

    try:
        summary = stabilize_folder(
            input_dir=args.input,
            output_dir=args.output,
            anchor=anchor,
            progress_cb=lambda v: emit({"type": "progress", "value": v}),
            log_cb=lambda m: emit({"type": "log", "msg": m}),
            smooth_radius=args.smooth,
            jpeg_quality=args.quality,
            debug_dir=args.debug_frames,
            border_mode=args.border_mode,
        )
        emit({"type": "done", "summary": summary})
    except Exception as exc:
        emit({"type": "error", "msg": str(exc)})
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Perforation stabilizer CLI")

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=["batch", "preview"],
        default="batch",
        help="Operation mode: batch (default) or preview",
    )

    # ── Batch-mode params ─────────────────────────────────────────────────────
    parser.add_argument("--input", default=None, help="Input folder with frames")
    parser.add_argument("--output", default=None, help="Output folder")
    parser.add_argument(
        "--anchor-x",
        type=float,
        default=None,
        help="Reference anchor X coordinate (required for batch)",
    )
    parser.add_argument(
        "--anchor-y",
        type=float,
        default=None,
        help="Reference anchor Y coordinate (required for batch)",
    )
    parser.add_argument(
        "--smooth", type=int, default=9, help="Moving-average radius (default 9)"
    )
    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG quality 1-100, 0=PNG (default 95)"
    )
    parser.add_argument(
        "--debug-frames",
        default=None,
        metavar="DIR",
        help="Optional folder for failed-frame debug JPEGs",
    )
    parser.add_argument(
        "--border-mode",
        choices=["replicate", "constant", "reflect"],
        default="replicate",
        help="Border fill mode for warpAffine (default: replicate)",
    )

    # ── Preview-mode params ───────────────────────────────────────────────────
    parser.add_argument(
        "--frame-path", default=None, help="Path to single frame for preview mode"
    )
    parser.add_argument(
        "--preview-out", default=None, help="Output path for preview JPEG"
    )

    args = parser.parse_args()

    if args.mode == "preview":
        if not args.frame_path or not args.preview_out:
            emit(
                {
                    "type": "error",
                    "msg": "--frame-path and --preview-out are required in preview mode",
                }
            )
            sys.exit(1)
        run_preview(args)
    else:
        if not args.input or not args.output:
            emit(
                {
                    "type": "error",
                    "msg": "--input and --output are required in batch mode",
                }
            )
            sys.exit(1)
        if args.anchor_x is None or args.anchor_y is None:
            emit(
                {
                    "type": "error",
                    "msg": "--anchor-x and --anchor-y are required in batch mode",
                }
            )
            sys.exit(1)
        run_batch(args)


if __name__ == "__main__":
    main()
