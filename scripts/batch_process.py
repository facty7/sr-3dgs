#!/usr/bin/env python3
"""Batch processing helper for multiple scenes.

Usage:
    # Process all directories under a parent folder
    python batch_process.py --parent_dir /data/scenes/

    # Process with specific config
    python batch_process.py --parent_dir /data/scenes/ --sr_model supir --sr_scale 4

    # Process only directories matching a pattern
    python batch_process.py --parent_dir /data/scenes/ --filter "scene_*"
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

_script_dir = Path(__file__).parent
_run_script = _script_dir / "run_pipeline.py"


def main():
    parser = argparse.ArgumentParser(
        description="Batch SR-3DGS processing for multiple scenes"
    )
    parser.add_argument("--parent_dir", type=str, required=True,
                        help="Parent directory containing multiple scene folders")
    parser.add_argument("--filter", type=str, default="*",
                        help="Glob pattern to filter scene directories")
    parser.add_argument("--work_dir", type=str, default="workspace_batch",
                        help="Root working directory")
    parser.add_argument("--sr_model", type=str, default="real-esrgan")
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=30_000)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel processes (uses CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--dry_run", action="store_true",
                        help="List scenes without processing")

    args = parser.parse_args()

    parent = Path(args.parent_dir)
    if not parent.exists():
        print(f"Error: {parent} does not exist")
        sys.exit(1)

    # Find all scene directories
    scenes = []
    for d in sorted(parent.glob(args.filter)):
        if d.is_dir():
            # Check if it has images
            images = list(d.glob("*")) + list(d.glob("images/*"))
            has_images = any(
                p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in images
            )
            if has_images:
                scenes.append(d)

    print(f"Found {len(scenes)} scenes in {parent}")
    for s in scenes:
        print(f"  - {s.name}")

    if args.dry_run:
        return

    # Process each scene
    for i, scene in enumerate(scenes):
        scene_work = Path(args.work_dir) / scene.name
        gpu = i % args.parallel if args.parallel > 1 else 0

        cmd = [
            sys.executable, str(_run_script),
            "--input_dir", str(scene),
            "--work_dir", str(scene_work),
            "--sr_model", args.sr_model,
            "--sr_scale", str(args.sr_scale),
            "--max_steps", str(args.max_steps),
        ]

        env = os.environ.copy()
        if args.parallel > 1:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        print(f"\nProcessing [{i+1}/{len(scenes)}]: {scene.name} (GPU {gpu})")
        subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    main()
