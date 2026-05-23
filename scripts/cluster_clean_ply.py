#!/usr/bin/env python3
"""Object-scene PLY cleanup using opacity, radius, and voxel components.

The goal is conservative cleanup for object captures: remove distant floaters
and detached background sheets while preserving a valid 3DGS PLY layout.
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}

CLEANUP_PRESETS = {
    "conservative": {
        "opacity_min": 0.10,
        "voxel_size": 0.12,
        "max_radius": 0.0,
        "max_scale": 0.0,
        "dilate": 1,
    },
    "balanced": {
        "opacity_min": 0.12,
        "voxel_size": 0.10,
        "max_radius": 0.0,
        "max_scale": 0.0,
        "dilate": 1,
    },
    "aggressive": {
        "opacity_min": 0.12,
        "voxel_size": 0.10,
        "max_radius": 0.0,
        "max_scale": 0.06,
        "dilate": 1,
    },
}


def _read_ply(path):
    path = Path(path)
    with path.open("rb") as f:
        header = []
        props = []
        vertex_count = None
        in_vertex = False
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"{path} ended before end_header")
            line = raw.decode("ascii").strip()
            header.append(line)
            if line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) != 3 or parts[1] not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property line: {line}")
                props.append((parts[2], np.dtype(PLY_TYPES[parts[1]])))
            elif line == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"{path} has no vertex element")
        dtype = np.dtype(props)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
    return props, data


def _write_ply(path, props, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inv = {np.dtype(v).str: k for k, v in PLY_TYPES.items()}
    with path.open("wb") as f:
        lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(data)}",
        ]
        for name, dtype in props:
            key = np.dtype(dtype).str
            if key not in inv:
                raise ValueError(f"Cannot write unsupported dtype {dtype} for {name}")
            lines.append(f"property {inv[key]} {name}")
        lines.append("end_header")
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        data.tofile(f)


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _opacity_actual(data):
    if "opacity" not in data.dtype.names:
        return np.ones(len(data), dtype=np.float64)
    raw = np.asarray(data["opacity"], dtype=np.float64)
    if np.nanmin(raw) < 0.0 or np.nanmax(raw) > 1.0:
        return _sigmoid(raw)
    return raw


def _xyz(data):
    for name in ("x", "y", "z"):
        if name not in data.dtype.names:
            raise ValueError(f"Missing required PLY field: {name}")
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)


def _scale_actual(data):
    names = data.dtype.names or ()
    if not all(name in names for name in ("scale_0", "scale_1", "scale_2")):
        return None
    scales = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float64)
    if np.nanmedian(scales) < -0.25:
        scales = np.exp(np.clip(scales, -20.0, 20.0))
    return scales


def _neighbor_offsets(connectivity):
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan != 1:
                    continue
                if connectivity == 18 and manhattan > 2:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def _components(voxels, counts, connectivity):
    occupied = {tuple(v) for v in voxels.tolist()}
    count_by_voxel = {tuple(v): int(c) for v, c in zip(voxels.tolist(), counts.tolist())}
    offsets = _neighbor_offsets(connectivity)
    comps = []
    while occupied:
        start = occupied.pop()
        q = deque([start])
        members = [start]
        point_count = count_by_voxel[start]
        while q:
            cur = q.popleft()
            for off in offsets:
                nxt = (cur[0] + off[0], cur[1] + off[1], cur[2] + off[2])
                if nxt in occupied:
                    occupied.remove(nxt)
                    q.append(nxt)
                    members.append(nxt)
                    point_count += count_by_voxel[nxt]
        comps.append({"voxels": members, "point_count": point_count})
    comps.sort(key=lambda c: c["point_count"], reverse=True)
    return comps


def _parse_vec3(text):
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError("seed must be x,y,z")
    return np.array(vals, dtype=np.float64)


def _auto_cleanup_params(xyz, opacity, scales, seed, opacity_min, max_radius, max_scale):
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
    if not finite.any():
        return seed, max_radius, max_scale, {"enabled": True, "warning": "no finite points"}

    opa_floor = max(float(opacity_min), float(np.percentile(opacity[finite], 35)))
    high = finite & (opacity >= opa_floor)
    if high.sum() < 1000:
        high = finite & (opacity >= opacity_min)
    if high.sum() < 1000:
        high = finite

    auto_seed = np.median(xyz[high], axis=0)
    dist = np.linalg.norm(xyz[high] - auto_seed[None, :], axis=1)
    # Use core percentiles instead of p99 so a small detached cluster cannot
    # stretch the object radius estimate.
    auto_radius = float(max(
        np.percentile(dist, 85) * 2.20,
        np.percentile(dist, 90) * 1.75,
        0.5,
    ))

    auto_scale = 0.0
    if scales is not None:
        scale_max = np.nanmax(scales[high], axis=1)
        # Large splats often appear as haze. Estimate from the object core and
        # let users override max_scale when they know a scene needs looser caps.
        auto_scale = float(max(
            np.percentile(scale_max, 90) * 1.80,
            np.percentile(scale_max, 95) * 1.20,
        ))

    resolved_seed = auto_seed if np.allclose(seed, 0.0) else seed
    resolved_radius = auto_radius if not max_radius or max_radius <= 0 else max_radius
    resolved_scale = auto_scale if scales is not None and (not max_scale or max_scale <= 0) else max_scale
    report = {
        "enabled": True,
        "high_confidence_count": int(high.sum()),
        "opacity_floor": float(opa_floor),
        "estimated_seed": [float(x) for x in auto_seed],
        "estimated_max_radius": float(auto_radius),
        "estimated_max_scale": float(auto_scale),
        "used_seed": [float(x) for x in resolved_seed],
        "used_max_radius": float(resolved_radius),
        "used_max_scale": float(resolved_scale),
    }
    return resolved_seed, resolved_radius, resolved_scale, report


def clean_ply(
    input_ply,
    output_ply,
    report_path=None,
    opacity_min=0.08,
    max_radius=0.0,
    max_scale=0.0,
    voxel_size=0.10,
    connectivity=26,
    select="nearest",
    seed=(0.0, 0.0, 0.0),
    min_component_points=1000,
    dilate=1,
    auto=False,
    preset="",
):
    base_defaults = {
        "opacity_min": 0.08,
        "voxel_size": 0.10,
        "max_radius": 0.0,
        "max_scale": 0.0,
        "dilate": 1,
    }
    if preset:
        if preset not in CLEANUP_PRESETS:
            raise ValueError(f"Unknown cleanup preset: {preset}")
        defaults = CLEANUP_PRESETS[preset]
    else:
        defaults = base_defaults

    opacity_min = defaults["opacity_min"] if opacity_min is None else opacity_min
    voxel_size = defaults["voxel_size"] if voxel_size is None else voxel_size
    max_radius = defaults["max_radius"] if max_radius is None else max_radius
    max_scale = defaults["max_scale"] if max_scale is None else max_scale
    dilate = defaults["dilate"] if dilate is None else dilate

    props, data = _read_ply(input_ply)
    xyz = _xyz(data)
    opacity = _opacity_actual(data)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
    base_mask = finite & (opacity >= opacity_min)

    seed = np.asarray(seed, dtype=np.float64)
    radius = np.linalg.norm(xyz - seed[None, :], axis=1)
    if max_radius and max_radius > 0:
        base_mask &= radius <= max_radius

    scales = _scale_actual(data)
    auto_report = {"enabled": False}
    if auto:
        seed, max_radius, max_scale, auto_report = _auto_cleanup_params(
            xyz, opacity, scales, seed, opacity_min, max_radius, max_scale
        )

    if max_scale and max_scale > 0 and scales is not None:
        base_mask &= np.nanmax(scales, axis=1) <= max_scale

    idx = np.flatnonzero(base_mask)
    if len(idx) == 0:
        raise ValueError("All points were removed before component cleanup")

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    voxel = np.floor((xyz[idx] - seed[None, :]) / voxel_size).astype(np.int64)
    unique_voxels, inv, counts = np.unique(voxel, axis=0, return_inverse=True, return_counts=True)
    comps = _components(unique_voxels, counts, connectivity)
    valid = [c for c in comps if c["point_count"] >= min_component_points]
    if not valid:
        valid = comps[:1]

    voxel_centers = (unique_voxels.astype(np.float64) + 0.5) * voxel_size + seed[None, :]
    center_by_voxel = {tuple(v): voxel_centers[i] for i, v in enumerate(unique_voxels.tolist())}

    def comp_center(comp):
        pts = np.array([center_by_voxel[v] for v in comp["voxels"]], dtype=np.float64)
        return np.mean(pts, axis=0)

    if select == "largest":
        keep_comp = valid[0]
    elif select == "nearest":
        keep_comp = min(valid, key=lambda c: float(np.linalg.norm(comp_center(c) - seed)))
    else:
        raise ValueError("select must be largest or nearest")

    keep_voxels = {tuple(v) for v in keep_comp["voxels"]}
    if dilate > 0:
        offsets = _neighbor_offsets(26)
        frontier = set(keep_voxels)
        for _ in range(dilate):
            expanded = set(frontier)
            for cur in frontier:
                for off in offsets:
                    expanded.add((cur[0] + off[0], cur[1] + off[1], cur[2] + off[2]))
            frontier = expanded
        keep_voxels = frontier

    kept_in_base = np.fromiter((tuple(v) in keep_voxels for v in voxel), dtype=bool, count=len(voxel))
    keep_mask = np.zeros(len(data), dtype=bool)
    keep_mask[idx] = kept_in_base
    cleaned = data[keep_mask]
    _write_ply(output_ply, props, cleaned)

    comp_summary = []
    for comp in comps[:12]:
        center = comp_center(comp)
        comp_summary.append({
            "point_count": int(comp["point_count"]),
            "voxel_count": int(len(comp["voxels"])),
            "center": [float(x) for x in center],
            "distance_to_seed": float(np.linalg.norm(center - seed)),
        })

    report = {
        "input": str(input_ply),
        "output": str(output_ply),
        "input_count": int(len(data)),
        "base_count": int(len(idx)),
        "output_count": int(len(cleaned)),
        "removed_count": int(len(data) - len(cleaned)),
        "removed_percent": float((len(data) - len(cleaned)) / max(1, len(data)) * 100.0),
        "opacity_min": float(opacity_min),
        "max_radius": float(max_radius),
        "max_scale": float(max_scale),
        "voxel_size": float(voxel_size),
        "connectivity": int(connectivity),
        "select": select,
        "seed": [float(x) for x in seed],
        "min_component_points": int(min_component_points),
        "dilate": int(dilate),
        "preset": preset,
        "auto": auto_report,
        "component_count": int(len(comps)),
        "selected_component": comp_summary[0] if select == "largest" else {
            "point_count": int(keep_comp["point_count"]),
            "voxel_count": int(len(keep_comp["voxels"])),
            "center": [float(x) for x in comp_center(keep_comp)],
            "distance_to_seed": float(np.linalg.norm(comp_center(keep_comp) - seed)),
        },
        "largest_components": comp_summary,
    }
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply")
    parser.add_argument("output_ply")
    parser.add_argument("--report", default="")
    parser.add_argument("--opacity_min", type=float, default=None)
    parser.add_argument("--max_radius", type=float, default=None)
    parser.add_argument("--max_scale", type=float, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument("--select", choices=["largest", "nearest"], default="nearest")
    parser.add_argument("--seed", default="0,0,0")
    parser.add_argument("--min_component_points", type=int, default=1000)
    parser.add_argument("--dilate", type=int, default=None)
    parser.add_argument("--preset", choices=sorted(CLEANUP_PRESETS), default="",
                        help="Cleanup preset. Explicit numeric args still override defaults.")
    parser.add_argument("--auto", action="store_true",
                        help="Estimate seed, radius, and scale limits from the PLY")
    args = parser.parse_args()

    report = clean_ply(
        args.input_ply,
        args.output_ply,
        report_path=args.report or None,
        opacity_min=args.opacity_min,
        max_radius=args.max_radius,
        max_scale=args.max_scale,
        voxel_size=args.voxel_size,
        connectivity=args.connectivity,
        select=args.select,
        seed=_parse_vec3(args.seed),
        min_component_points=args.min_component_points,
        dilate=None if args.dilate is None else max(0, args.dilate),
        auto=args.auto,
        preset=args.preset,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
