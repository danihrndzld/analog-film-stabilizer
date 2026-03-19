#!/usr/bin/env python3
"""
CLI wrapper for stabilize_folder.
Streams JSON-lines to stdout so Electron can consume progress and logs.

Usage:
  python3 stabilizer_cli.py --input DIR --output DIR [--roi F] [--threshold N] [--smooth N] [--quality N]

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
import types

# ── Tkinter mock ──────────────────────────────────────────────────────────────
# perforation_stabilizer_app.py imports tkinter at module level for its GUI.
# We only need the pure processing functions (stabilize_folder and helpers),
# so we inject stub modules before importing to avoid requiring a Tk build.
class _Stub:
    """A stub that silently absorbs any attribute access or call."""
    def __getattr__(self, _):    return _Stub()
    def __call__(self, *a, **kw): return _Stub()
    def __iter__(self):           return iter([])
    def __bool__(self):           return False

class _StubModule(types.ModuleType):
    """Module subclass that returns _Stub() for any missing attribute."""
    def __getattr__(self, attr):
        return _Stub()

def _make_stub(name):
    return _StubModule(name)

for _mod_name in ['tkinter', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.ttk', 'tkinterdnd2']:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _make_stub(_mod_name)

# Locate perforation_stabilizer_app.py next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perforation_stabilizer_app import stabilize_folder


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Perforation stabilizer CLI")
    parser.add_argument("--input",     required=True,              help="Input folder with frames")
    parser.add_argument("--output",    required=True,              help="Output folder")
    parser.add_argument("--roi",       type=float, default=0.22,   help="Left ROI fraction (default 0.22)")
    parser.add_argument("--threshold", type=int,   default=210,    help="Brightness threshold (default 210)")
    parser.add_argument("--smooth",    type=int,   default=9,      help="Moving-average radius (default 9)")
    parser.add_argument("--quality",   type=int,   default=95,     help="JPEG quality 1-100, 0=PNG (default 95)")
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
        )
        emit({"type": "done", "summary": summary})
    except Exception as exc:
        emit({"type": "error", "msg": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
