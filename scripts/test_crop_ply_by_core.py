#!/usr/bin/env python3
"""Smoke test for high-confidence core PLY cropping."""

import tempfile
from pathlib import Path

import numpy as np

from cluster_clean_ply import _write_ply
from crop_ply_by_core import crop_ply


def _make_ply(path):
    rng = np.random.default_rng(7)
    core = rng.normal(0.0, 0.18, size=(1000, 3)).astype(np.float32)
    tail = np.column_stack([
        rng.normal(1.8, 0.03, size=80),
        rng.normal(0.0, 0.03, size=80),
        rng.normal(0.0, 0.03, size=80),
    ]).astype(np.float32)
    xyz = np.vstack([core, tail])
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
    data["opacity"][: len(core)] = 2.0
    data["opacity"][len(core):] = -2.0
    data["scale_0"] = data["scale_1"] = data["scale_2"] = np.log(0.02)
    _write_ply(path, [(name, data.dtype[name]) for name in data.dtype.names], data)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.ply"
        dst = Path(tmp) / "dst.ply"
        _make_ply(src)
        report = crop_ply(
            src,
            dst,
            core_opacity_percentile=70,
            lower_percentile=0.5,
            upper_percentile=99.5,
            margin=0.08,
            min_retention=0.7,
        )
        assert dst.exists(), report
        assert report["output_count"] < report["input_count"], report
        assert report["retention_percent"] > 80, report
        assert report["bbox_max"][0] < 1.8, report


if __name__ == "__main__":
    main()
