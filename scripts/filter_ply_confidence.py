#!/usr/bin/env python3
"""Filter low-confidence or haze-like Gaussians from a 3DGS PLY."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cluster_clean_ply import _opacity_actual, _read_ply, _scale_actual, _write_ply, _xyz


def _percentiles(values, points=(1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    values = np.asarray(values)
    return {str(p): float(np.percentile(values, p)) for p in points}


def filter_ply(
    input_ply,
    output_ply,
    report_path=None,
    opacity_min=0.0,
    max_scale=0.0,
    radius_percentile=0.0,
    radius_margin=0.0,
    min_retention=0.70,
):
    props, data = _read_ply(input_ply)
    xyz = _xyz(data)
    opacity = _opacity_actual(data)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
    keep = finite.copy()

    if opacity_min and opacity_min > 0:
        keep &= opacity >= opacity_min

    scales = _scale_actual(data)
    scale_max = None
    if scales is not None:
        scale_max = np.nanmax(scales, axis=1)
        if max_scale and max_scale > 0:
            keep &= np.isfinite(scale_max) & (scale_max <= max_scale)

    radius_limit = 0.0
    radius = None
    if radius_percentile and radius_percentile > 0:
        center = np.nanmedian(xyz[finite], axis=0)
        radius = np.linalg.norm(xyz - center[None, :], axis=1)
        radius_limit = float(np.percentile(radius[finite], radius_percentile) + radius_margin)
        keep &= radius <= radius_limit

    retention = float(keep.sum() / max(1, len(data)))
    if retention < min_retention:
        raise ValueError(
            f"filter would keep only {retention:.1%}; lower thresholds or min_retention"
        )

    filtered = data[keep]
    _write_ply(output_ply, props, filtered)
    report = {
        "input": str(input_ply),
        "output": str(output_ply),
        "input_count": int(len(data)),
        "output_count": int(len(filtered)),
        "removed_count": int(len(data) - len(filtered)),
        "removed_percent": float((len(data) - len(filtered)) / max(1, len(data)) * 100.0),
        "retention_percent": float(retention * 100.0),
        "opacity_min": float(opacity_min),
        "max_scale": float(max_scale),
        "radius_percentile": float(radius_percentile),
        "radius_margin": float(radius_margin),
        "radius_limit": float(radius_limit),
        "input_opacity_actual_percentiles": _percentiles(opacity[finite]),
    }
    if scale_max is not None:
        report["input_scale_actual_max_axis_percentiles"] = _percentiles(scale_max[finite])
    if radius is not None:
        report["input_radius_percentiles"] = _percentiles(radius[finite])
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
    parser.add_argument("--opacity_min", type=float, default=0.0)
    parser.add_argument("--max_scale", type=float, default=0.0)
    parser.add_argument("--radius_percentile", type=float, default=0.0)
    parser.add_argument("--radius_margin", type=float, default=0.0)
    parser.add_argument("--min_retention", type=float, default=0.70)
    args = parser.parse_args()
    report = filter_ply(
        args.input_ply,
        args.output_ply,
        report_path=args.report or None,
        opacity_min=args.opacity_min,
        max_scale=args.max_scale,
        radius_percentile=args.radius_percentile,
        radius_margin=args.radius_margin,
        min_retention=args.min_retention,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
