#!/usr/bin/env python3
"""Synthetic tests for candidate comparison scale normalization."""

import json
import tempfile
from pathlib import Path

import numpy as np

from compare_clean_candidates import compare


PROPS = [
    ("x", np.dtype("<f4")),
    ("y", np.dtype("<f4")),
    ("z", np.dtype("<f4")),
    ("opacity", np.dtype("<f4")),
    ("scale_0", np.dtype("<f4")),
    ("scale_1", np.dtype("<f4")),
    ("scale_2", np.dtype("<f4")),
]


def _write_ply(path, scale_values):
    path = Path(path)
    n = len(scale_values)
    data = np.zeros(n, dtype=np.dtype(PROPS))
    data["x"] = np.linspace(-0.2, 0.2, n)
    data["y"] = np.sin(np.linspace(0.0, 1.0, n)) * 0.1
    data["z"] = np.cos(np.linspace(0.0, 1.0, n)) * 0.1
    data["opacity"] = 0.8
    data["scale_0"] = scale_values
    data["scale_1"] = scale_values
    data["scale_2"] = scale_values
    with path.open("wb") as f:
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        header.extend(f"property float {name}" for name, _ in PROPS)
        header.append("end_header")
        f.write(("\n".join(header) + "\n").encode("ascii"))
        data.tofile(f)


def run_test(tmp_dir):
    tmp = Path(tmp_dir)
    actual_scale = np.full(128, 0.05, dtype=np.float32)
    log_scale = np.log(actual_scale).astype(np.float32)
    base = tmp / "base_actual.ply"
    candidate = tmp / "candidate_log.ply"
    _write_ply(base, actual_scale)
    _write_ply(candidate, log_scale)

    result = compare(base, [candidate])
    delta = result["candidates"][0]["delta_from_base"]
    ok = (
        result["ok"]
        and result["base"]["scale_kind"] == "actual"
        and result["candidates"][0]["scale_kind"] == "log"
        and abs(delta["scale_actual_p99_delta"]) < 1e-5
        and result["recommended_candidate"]["path"].endswith("candidate_log.ply")
    )
    return {"ok": bool(ok), "result": result}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        result = run_test(tmp)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
