#!/usr/bin/env python3
"""Smoke test for confidence-based PLY filtering."""

import tempfile
from pathlib import Path

import numpy as np

from cluster_clean_ply import _read_ply, _write_ply
from filter_ply_confidence import filter_ply


def _make_ply(path):
    rng = np.random.default_rng(11)
    good = rng.normal(0.0, 0.15, size=(1000, 3)).astype(np.float32)
    haze = rng.normal(0.0, 0.55, size=(120, 3)).astype(np.float32)
    xyz = np.vstack([good, haze])
    dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
    ])
    data = np.zeros(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["opacity"][: len(good)] = 1.0
    data["opacity"][len(good):] = -1.0
    data["scale_0"][: len(good)] = np.log(0.02)
    data["scale_1"][: len(good)] = np.log(0.02)
    data["scale_2"][: len(good)] = np.log(0.02)
    data["scale_0"][len(good):] = np.log(0.09)
    data["scale_1"][len(good):] = np.log(0.09)
    data["scale_2"][len(good):] = np.log(0.09)
    _write_ply(path, [(name, data.dtype[name]) for name in data.dtype.names], data)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.ply"
        dst = Path(tmp) / "dst.ply"
        _make_ply(src)
        report = filter_ply(
            src,
            dst,
            opacity_min=0.5,
            max_scale=0.04,
            min_retention=0.80,
        )
        _, data = _read_ply(dst)
        assert dst.exists(), report
        assert report["output_count"] == len(data), report
        assert report["removed_count"] == 120, report
        assert report["retention_percent"] > 85.0, report


if __name__ == "__main__":
    main()
