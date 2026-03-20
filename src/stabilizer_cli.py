#!/usr/bin/env python3
"""
CLI backend for the Electron UI.
Streams JSON-lines to stdout so Electron can consume progress and logs.

Usage:
  python3 stabilizer_cli.py --input DIR --output DIR [--roi F] [--threshold N]
                             [--smooth N] [--quality N] [--film-format super8|8mm|super16]

Output lines (one per line, each valid JSON):
  {"type": "progress", "value": 0.0..1.0}
  {"type": "log",      "msg":  "..."}
  {"type": "done",     "summary": {...}}
  {"type": "error",    "msg":  "..."}
"""

import sys
import json
import argparse
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perforation_stabilizer_app import stabilize_folder


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Perforation stabilizer CLI")
    parser.add_argument("--input",       required=True,                                   help="Input folder with frames")
    parser.add_argument("--output",      required=True,                                   help="Output folder")
    parser.add_argument("--roi",         type=float, default=0.22,                        help="ROI fraction (default 0.22)")
    parser.add_argument("--threshold",   type=int,   default=210,                         help="Brightness threshold (default 210)")
    parser.add_argument("--smooth",      type=int,   default=9,                           help="Moving-average radius (default 9)")
    parser.add_argument("--quality",     type=int,   default=95,                          help="JPEG quality 1-100, 0=PNG (default 95)")
    parser.add_argument("--film-format", choices=["super8", "8mm", "super16"], default="super8",
                        help="Film format: super8 (default), 8mm, super16")
    args = parser.parse_args()

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
        )
        emit({"type": "done", "summary": summary})
    except Exception as exc:
        emit({"type": "error", "msg": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
