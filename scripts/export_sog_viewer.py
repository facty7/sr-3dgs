#!/usr/bin/env python3
"""CLI wrapper for PlayCanvas SOG viewer export."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.sog_export import export_sog_viewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply")
    parser.add_argument("output_html")
    parser.add_argument("--bundled", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()
    export_sog_viewer(
        args.input_ply,
        args.output_html,
        overwrite=True,
        unbundled=not args.bundled,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
