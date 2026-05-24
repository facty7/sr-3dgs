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


def _parse_extraction_variant(text):
    parts = text.split(":")
    if len(parts) not in (3, 4, 5, 6):
        raise argparse.ArgumentTypeError(
            "extraction variant must be name:min_frames:max_frames[:fps][:span][:adaptive|strict]"
        )
    name, min_frames, max_frames, *rest = parts
    label = _safe_label(name)
    if not label:
        raise argparse.ArgumentTypeError("extraction variant name cannot be empty")
    try:
        min_frames = int(min_frames)
        max_frames = int(max_frames)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("min_frames and max_frames must be integers") from exc
    if min_frames <= 0 or max_frames <= 0 or max_frames < min_frames:
        raise argparse.ArgumentTypeError("extraction frame counts must satisfy 0 < min <= max")

    fps = None
    min_span = None
    adaptive = True
    for item in rest:
        lowered = item.lower()
        if lowered in {"adaptive", "strict"}:
            adaptive = lowered == "adaptive"
            continue
        explicit_fps = False
        explicit_span = False
        if lowered.startswith("span"):
            explicit_span = True
            if min_span is not None:
                raise argparse.ArgumentTypeError("only one span value is allowed")
            lowered = lowered[4:]
        elif lowered.startswith("fps"):
            explicit_fps = True
            if fps is not None:
                raise argparse.ArgumentTypeError("only one fps value is allowed")
            lowered = lowered[3:]
        try:
            value = float(lowered if lowered else item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "optional extraction value must be fps, span, adaptive, or strict"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError("fps/span values must be positive")
        if explicit_span:
            if value > 1.0:
                raise argparse.ArgumentTypeError("span must be <= 1.0")
            min_span = value
        elif explicit_fps:
            fps = value
        elif value <= 1.0:
            if min_span is not None:
                raise argparse.ArgumentTypeError("only one span value is allowed")
            min_span = value
        elif fps is None:
            fps = value
        else:
            raise argparse.ArgumentTypeError("only one fps and one span value are allowed")

    return {
        "name": label,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "fps": fps,
        "min_span": min_span,
        "adaptive": adaptive,
    }


def _safe_label(text):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(text))


def _scene_name(video_path, output_name):
    if output_name:
        base = output_name
    else:
        base = Path(video_path).stem
    return _safe_label(base)


def _build_command(args, strategy, extraction_variant=None):
    suffix = f"{strategy['mode']}_x{strategy['scale']}"
    if strategy["mode"] == "model":
        suffix = f"{strategy['mode']}_{strategy['model'].replace('+', 'p')}_x{strategy['scale']}"
    if extraction_variant:
        suffix = f"{extraction_variant['name']}_{suffix}"
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
    if extraction_variant:
        cmd.extend(["--extract_min_frames", str(extraction_variant["min_frames"])])
        cmd.extend(["--extract_max_frames", str(extraction_variant["max_frames"])])
        if extraction_variant.get("fps") is not None:
            cmd.extend(["--extract_fps", str(extraction_variant["fps"])])
        if extraction_variant.get("min_span") is not None:
            cmd.extend(["--extract_min_span", str(extraction_variant["min_span"])])
        if extraction_variant.get("adaptive") is False:
            cmd.append("--no_adaptive_extract")
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


def _summary_command(plan_path, report_path):
    return [
        sys.executable,
        "scripts/summarize_sr_sweep.py",
        "--plan",
        str(plan_path),
        "--report",
        str(report_path),
    ]


def _default_extraction_variants():
    return [
        {"name": "cover64", "min_frames": 64, "max_frames": 200, "fps": None, "min_span": 0.85, "adaptive": True},
        {"name": "cover96", "min_frames": 96, "max_frames": 300, "fps": None, "min_span": 0.90, "adaptive": True},
        {"name": "strict64", "min_frames": 64, "max_frames": 200, "fps": None, "min_span": 0.85, "adaptive": False},
    ]


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
    parser.add_argument(
        "--extract_variant",
        action="append",
        type=_parse_extraction_variant,
        help=(
            "Extraction sweep variant as name:min_frames:max_frames[:fps][:span][:adaptive|strict]. "
            "Values <= 1.0 are treated as min timeline span unless prefixed with fps. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--phone_coverage_sweep",
        action="store_true",
        help="Cross SR strategies with cover64, cover96, and strict64 extraction variants.",
    )
    parser.add_argument("--plan", default="")
    parser.add_argument(
        "--summary_report",
        default="",
        help="Summary JSON path written by the generated summarize command.",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    strategies = args.strategy or [
        {"mode": "off", "model": "real-esrgan", "scale": 1},
        {"mode": "resize", "model": "real-esrgan", "scale": 2},
        {"mode": "auto", "model": "real-esrgan", "scale": 2},
        {"mode": "model", "model": "real-esrgan", "scale": 2},
    ]
    extraction_variants = list(args.extract_variant or [])
    if args.phone_coverage_sweep:
        extraction_variants = _default_extraction_variants() + extraction_variants
    if not extraction_variants:
        extraction_variants = [None]

    plan_path = Path(args.plan) if args.plan else Path(args.work_dir) / "sr_sweep_plan.json"
    report_path = (
        Path(args.summary_report)
        if args.summary_report
        else Path(args.work_dir) / "sr_sweep_summary.json"
    )

    plan = {
        "video": args.video,
        "preset": args.preset,
        "work_root": args.work_dir,
        "final_output_root": args.final_output_dir,
        "phone_coverage_sweep": bool(args.phone_coverage_sweep),
        "summary_report": str(report_path),
        "summary_command": _summary_command(plan_path, report_path),
        "runs": [],
    }
    for extraction_variant in extraction_variants:
        for strategy in strategies:
            output_name, cmd = _build_command(args, strategy, extraction_variant)
            plan["runs"].append({
                "output_name": output_name,
                "workspace_dir": str(Path(args.work_dir) / output_name),
                "output_dir": str(Path(args.final_output_dir) / output_name),
                "strategy": strategy,
                "extraction_variant": extraction_variant or {},
                "command": cmd,
            })

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote plan: {plan_path}")
    print("\nSummary command:")
    print(" ".join(plan["summary_command"]))
    for run in plan["runs"]:
        print("\n" + " ".join(run["command"]))

    if args.run:
        for run in plan["runs"]:
            print(f"\n[RUN] {run['output_name']}")
            subprocess.run(run["command"], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
