#!/usr/bin/env python3
"""Crop a 3DGS PLY to a high-confidence object core bounding box."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cluster_clean_ply import _opacity_actual, _read_ply, _scale_actual, _write_ply, _xyz


def crop_ply(
    input_ply,
    output_ply,
    report_path=None,
    core_opacity_percentile=80.0,
    lower_percentile=0.5,
    upper_percentile=99.5,
    margin=0.12,
    max_scale=0.0,
    min_retention=0.65,
):
    props, data = _read_ply(input_ply)
    xyz = _xyz(data)
    opacity = _opacity_actual(data)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
    core = finite & (opacity >= np.percentile(opacity[finite], core_opacity_percentile))

    scales = _scale_actual(data)
    if max_scale and max_scale > 0 and scales is not None:
        scale_ok = np.nanmax(scales, axis=1) <= max_scale
        core &= scale_ok
    if core.sum() < 1000:
        raise ValueError("core selection is too small; relax core_opacity_percentile or max_scale")

    lo = np.percentile(xyz[core], lower_percentile, axis=0) - margin
    hi = np.percentile(xyz[core], upper_percentile, axis=0) + margin
    keep = finite & np.all((xyz >= lo[None, :]) & (xyz <= hi[None, :]), axis=1)
    retention = keep.sum() / max(1, len(data))
    if retention < min_retention:
        raise ValueError(
            f"crop would keep only {retention:.1%}; lower min_retention or use a larger margin"
        )

    cropped = data[keep]
    _write_ply(output_ply, props, cropped)
    report = {
        "input": str(input_ply),
        "output": str(output_ply),
        "input_count": int(len(data)),
        "core_count": int(core.sum()),
        "output_count": int(len(cropped)),
        "removed_count": int(len(data) - len(cropped)),
        "removed_percent": float((len(data) - len(cropped)) / max(1, len(data)) * 100.0),
        "retention_percent": float(retention * 100.0),
        "core_opacity_percentile": float(core_opacity_percentile),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "margin": float(margin),
        "max_scale": float(max_scale),
        "bbox_min": [float(x) for x in lo],
        "bbox_max": [float(x) for x in hi],
    }
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply")
    parser.add_argument("output_ply")
    parser.add_argument("--report", default="")
    parser.add_argument("--core_opacity_percentile", type=float, default=80.0)
    parser.add_argument("--lower_percentile", type=float, default=0.5)
    parser.add_argument("--upper_percentile", type=float, default=99.5)
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--max_scale", type=float, default=0.0)
    parser.add_argument("--min_retention", type=float, default=0.65)
    args = parser.parse_args()
    result = crop_ply(
        args.input_ply,
        args.output_ply,
        report_path=args.report or None,
        core_opacity_percentile=args.core_opacity_percentile,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        margin=args.margin,
        max_scale=args.max_scale,
        min_retention=args.min_retention,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
