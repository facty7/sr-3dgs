#!/usr/bin/env python3
"""Small synthetic tests for cluster_clean_ply auto mode."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from cluster_clean_ply import clean_ply


PROPS = [
    ("x", np.dtype("<f4")),
    ("y", np.dtype("<f4")),
    ("z", np.dtype("<f4")),
    ("opacity", np.dtype("<f4")),
    ("scale_0", np.dtype("<f4")),
    ("scale_1", np.dtype("<f4")),
    ("scale_2", np.dtype("<f4")),
]


def _write_ply(path, data):
    path = Path(path)
    with path.open("wb") as f:
        lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(data)}",
        ]
        for name, _ in PROPS:
            lines.append(f"property float {name}")
        lines.append("end_header")
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        data.tofile(f)


def _make_scene(seed=7):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(PROPS)

    body = rng.normal(loc=(0.1, 0.2, -0.1), scale=(0.22, 0.24, 0.20), size=(2400, 3))
    body_scale = rng.uniform(0.006, 0.035, size=(len(body), 3))
    body_opacity = rng.uniform(0.35, 0.95, size=(len(body), 1))

    floaters = rng.normal(loc=(4.0, 3.5, -3.0), scale=(0.2, 0.2, 0.2), size=(160, 3))
    floater_scale = rng.uniform(0.006, 0.03, size=(len(floaters), 3))
    floater_opacity = rng.uniform(0.45, 0.90, size=(len(floaters), 1))

    haze = rng.normal(loc=(0.1, 0.2, -0.1), scale=(0.28, 0.28, 0.28), size=(80, 3))
    haze_scale = rng.uniform(0.20, 0.35, size=(len(haze), 3))
    haze_opacity = rng.uniform(0.40, 0.80, size=(len(haze), 1))

    xyz = np.vstack([body, floaters, haze])
    opacity = np.vstack([body_opacity, floater_opacity, haze_opacity])
    scales = np.vstack([body_scale, floater_scale, haze_scale])

    data = np.zeros(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["opacity"] = opacity[:, 0]
    data["scale_0"], data["scale_1"], data["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    return data


def run_test(tmp_dir):
    tmp = Path(tmp_dir)
    src = tmp / "synthetic.ply"
    dst = tmp / "synthetic_clean.ply"
    report = tmp / "report.json"
    data = _make_scene()
    _write_ply(src, data)
    meta = clean_ply(
        src,
        dst,
        report_path=report,
        opacity_min=0.10,
        voxel_size=0.12,
        max_radius=0.0,
        max_scale=0.0,
        select="nearest",
        seed=(0.0, 0.0, 0.0),
        min_component_points=100,
        dilate=1,
        auto=True,
    )
    aggressive_dst = tmp / "synthetic_aggressive.ply"
    aggressive_meta = clean_ply(
        src,
        aggressive_dst,
        opacity_min=None,
        voxel_size=None,
        max_radius=None,
        max_scale=None,
        select="nearest",
        seed=(0.0, 0.0, 0.0),
        min_component_points=100,
        dilate=None,
        auto=True,
        preset="aggressive",
    )
    ok = (
        dst.exists()
        and 2100 <= meta["output_count"] <= 2450
        and meta["removed_count"] >= 180
        and meta["auto"]["enabled"]
        and meta["auto"]["used_max_radius"] > 0
        and meta["auto"]["used_max_scale"] > 0
        and aggressive_meta["preset"] == "aggressive"
        and aggressive_meta["max_scale"] <= 0.06
        and aggressive_meta["output_count"] <= meta["output_count"]
    )
    return {"ok": bool(ok), "meta": meta, "aggressive_meta": aggressive_meta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp_dir", default="")
    args = parser.parse_args()
    if args.tmp_dir:
        result = run_test(args.tmp_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_test(tmp)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
