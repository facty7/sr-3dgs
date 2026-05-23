#!/usr/bin/env python3
"""Inspect 3DGS outputs for obvious viewer-breaking failures."""

import argparse
import struct
from pathlib import Path

import numpy as np


def read_splat(path):
    raw = Path(path).read_bytes()
    count = len(raw) // 32
    rows = np.empty((count, 14), dtype=np.float64)
    for i in range(count):
        off = i * 32
        rows[i, :6] = struct.unpack_from("<ffffff", raw, off)
        rows[i, 6:] = struct.unpack_from("<BBBBBBBB", raw, off + 24)
    return rows


def read_ply(path):
    with open(path, "rb") as f:
        props = []
        count = 0
        while True:
            line = f.readline().decode("utf-8").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float") or line.startswith("property uchar"):
                _, typ, name = line.split()
                props.append((name, {"float": np.float32, "uchar": np.uint8}[typ]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=count)


def summarize_xyz(name, xyz):
    center = np.nanmedian(xyz, axis=0)
    dist = np.linalg.norm(xyz - center, axis=1)
    print(f"{name}: {len(xyz):,} points")
    print(f"  xyz min: {np.nanmin(xyz, axis=0)}")
    print(f"  xyz max: {np.nanmax(xyz, axis=0)}")
    print(f"  radius p50/p95/p99/max: {np.percentile(dist, [50, 95, 99, 100])}")
    if np.isnan(xyz).any() or np.isinf(xyz).any():
        print("  FAIL: contains NaN or Inf coordinates")
    elif np.percentile(dist, 99) > 100:
        print("  WARN: p99 radius is very large; viewer framing may be poor")
    elif dist.max() > max(100, np.percentile(dist, 99) * 20):
        print("  WARN: extreme coordinate outliers found")
    else:
        print("  OK: coordinate range looks sane")


def diagnose(path):
    path = Path(path)
    if path.suffix.lower() == ".splat":
        rows = read_splat(path)
        summarize_xyz(str(path), rows[:, :3])
        scales = rows[:, 3:6]
        rgba = rows[:, 6:10]
        print(f"  scale p50/p95/p99/max: {np.percentile(scales.reshape(-1), [50, 95, 99, 100])}")
        print(f"  alpha p50/p95/max: {np.percentile(rgba[:, 3], [50, 95, 100])}")
        return

    if path.suffix.lower() == ".ply":
        data = read_ply(path)
        xyz = np.column_stack([data["x"], data["y"], data["z"]])
        summarize_xyz(str(path), xyz)
        if "scale_0" in data.dtype.names:
            scales = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]])
            print(f"  scale p50/p95/p99/max: {np.percentile(scales.reshape(-1), [50, 95, 99, 100])}")
        if "opacity" in data.dtype.names:
            opacity = data["opacity"]
            print(f"  opacity p50/p95/max: {np.percentile(opacity, [50, 95, 100])}")
        return

    raise ValueError(f"Unsupported file type: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="PLY or .splat files to inspect")
    args = parser.parse_args()
    for item in args.paths:
        diagnose(item)


if __name__ == "__main__":
    main()
