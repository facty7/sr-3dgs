#!/usr/bin/env python3
"""Estimate whether a requested heavy pipeline step should be run now."""

import argparse
import json
import shutil
from pathlib import Path


def _size_mb(path):
    path = Path(path)
    return round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0.0


def _disk_free_gb(path):
    usage = shutil.disk_usage(path)
    return round(usage.free / (1024 ** 3), 2)


def preflight(input_path, step, min_free_gb=5.0):
    input_path = Path(input_path)
    free_gb = _disk_free_gb(input_path.parent if input_path.exists() else Path.cwd())
    size_mb = _size_mb(input_path)
    warnings = []
    if not input_path.exists():
        warnings.append(f"missing input: {input_path}")
    if free_gb < min_free_gb:
        warnings.append(f"low disk space: {free_gb}GB free < {min_free_gb}GB")
    if step == "sog" and size_mb > 200:
        warnings.append(f"large PLY for SOG conversion: {size_mb}MB")
    if step == "render":
        warnings.append("browser/WebGL render QA can use significant GPU/CPU")
    if step == "train":
        warnings.append("training can saturate GPU/CPU for a long time")

    return {
        "ok": input_path.exists() and free_gb >= min_free_gb,
        "step": step,
        "input": str(input_path),
        "input_mb": size_mb,
        "free_disk_gb": free_gb,
        "min_free_disk_gb": min_free_gb,
        "warnings": warnings,
        "notes": [
            "This preflight is a lightweight guard, not a guarantee.",
            "Heavy steps still require explicit command-line opt-in in this project.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--step", choices=["sog", "render", "train"], required=True)
    parser.add_argument("--min_free_gb", type=float, default=5.0)
    args = parser.parse_args()
    result = preflight(args.input_path, args.step, args.min_free_gb)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
