#!/usr/bin/env python3
"""Create reproducible command plans for SR strategy sweeps.

By default this script only writes a plan JSON and prints commands. The --run
flag launches the expensive reconstructions.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _parse_strategy(text):
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "strategy must be mode:scale or mode:model:scale"
        )
    if len(parts) == 2:
        mode, scale = parts
        model = "real-esrgan"
    else:
        mode, model, scale = parts
    if mode not in {"auto", "off", "resize", "model"}:
        raise argparse.ArgumentTypeError(f"unsupported mode: {mode}")
    return {"mode": mode, "model": model, "scale": int(scale)}


def _scene_name(video_path, output_name):
    if output_name:
        base = output_name
    else:
        base = Path(video_path).stem
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in base)


def _build_command(args, strategy):
    suffix = f"{strategy['mode']}_x{strategy['scale']}"
    if strategy["mode"] == "model":
        suffix = f"{strategy['mode']}_{strategy['model'].replace('+', 'p')}_x{strategy['scale']}"
    output_name = f"{_scene_name(args.video, args.output_name)}_{suffix}"
    cmd = [
        sys.executable,
        "scripts/run_video_pipeline.py",
        "--video", args.video,
        "--output_name", output_name,
        "--work_dir", args.work_dir,
        "--final_output_dir", args.final_output_dir,
        "--preset", args.preset,
        "--sr_mode", strategy["mode"],
        "--sr_model", strategy["model"],
        "--sr_scale", str(strategy["scale"]),
    ]
    if args.projection:
        cmd.extend(["--projection", args.projection])
    if args.object_mask:
        cmd.extend(["--object_mask", args.object_mask])
    if args.object_bbox:
        cmd.extend(["--object_bbox", args.object_bbox])
    if args.cluster_clean:
        cmd.append("--cluster_clean")
    if args.no_showcase:
        cmd.append("--no_showcase")
    if args.extra_args:
        cmd.extend(args.extra_args)
    return output_name, cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_name", default="")
    parser.add_argument("--work_dir", default="workspace_video/sr_sweeps")
    parser.add_argument("--final_output_dir", default="output/sr_sweeps")
    parser.add_argument("--preset", default="standard")
    parser.add_argument("--projection", default="perspective")
    parser.add_argument("--object_mask", default="auto", choices=["off", "auto"])
    parser.add_argument("--object_bbox", default="")
    parser.add_argument("--cluster_clean", action="store_true")
    parser.add_argument("--no_showcase", action="store_true")
    parser.add_argument(
        "--strategy",
        action="append",
        type=_parse_strategy,
        help="mode:scale or mode:model:scale. Repeatable.",
    )
    parser.add_argument("--plan", default="")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    strategies = args.strategy or [
        {"mode": "off", "model": "real-esrgan", "scale": 1},
        {"mode": "resize", "model": "real-esrgan", "scale": 2},
        {"mode": "auto", "model": "real-esrgan", "scale": 2},
        {"mode": "model", "model": "real-esrgan", "scale": 2},
    ]
    plan = {
        "video": args.video,
        "preset": args.preset,
        "runs": [],
    }
    for strategy in strategies:
        output_name, cmd = _build_command(args, strategy)
        plan["runs"].append({
            "output_name": output_name,
            "strategy": strategy,
            "command": cmd,
        })

    plan_path = Path(args.plan) if args.plan else Path(args.work_dir) / "sr_sweep_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote plan: {plan_path}")
    for run in plan["runs"]:
        print("\n" + " ".join(run["command"]))

    if args.run:
        for run in plan["runs"]:
            print(f"\n[RUN] {run['output_name']}")
            subprocess.run(run["command"], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
