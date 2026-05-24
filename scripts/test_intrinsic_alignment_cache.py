#!/usr/bin/env python3
"""Unit tests for intrinsic-alignment cache invalidation."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sr_3dgs.step3_intrinsic_align import IntrinsicAligner


def _write_image(path: Path, width: int, height: int):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    arr[:, :, 1] = 128
    Image.fromarray(arr).save(path)


class FakeAligner(IntrinsicAligner):
    def _load_colmap_data(self):
        cameras = {
            1: {
                "model_id": 1,
                "model_name": "PINHOLE",
                "width": 24,
                "height": 16,
                "params": np.array([20.0, 20.0, 12.0, 8.0], dtype=np.float64),
            }
        }
        images = {
            1: {
                "qvec": np.array([1.0, 0.0, 0.0, 0.0]),
                "tvec": np.array([0.0, 0.0, 0.0]),
                "camtoworld": np.eye(4),
                "camera_id": 1,
                "name": "frame_000.png",
            }
        }
        points = {
            1: {
                "xyz": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "rgb": np.array([255, 255, 255], dtype=np.uint8),
            }
        }
        return cameras, images, points


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sparse = root / "sparse" / "0"
        sparse.mkdir(parents=True)
        (sparse / "cameras.bin").write_bytes(b"camera-v1")
        sr = root / "sr_images"
        sr.mkdir()
        aligned = root / "aligned"

        _write_image(sr / "frame_000.png", 24, 16)
        (sr / "sr_manifest.json").write_text(
            json.dumps({"effective_mode": "off", "effective_scale": [1.0, 1.0]}),
            encoding="utf-8",
        )
        aligner = FakeAligner(str(sparse), str(sr), str(aligned), scale_factor=4)
        aligner.run(force=False)
        meta1 = json.loads((aligned / "metadata.json").read_text(encoding="utf-8"))
        assert meta1["scale_factor_w"] == 1.0, meta1
        assert aligner._already_done() is True

        _write_image(sr / "frame_000.png", 48, 32)
        (sr / "sr_manifest.json").write_text(
            json.dumps({"effective_mode": "resize", "effective_scale": [2.0, 2.0]}),
            encoding="utf-8",
        )
        assert aligner._already_done() is False
        aligner.run(force=False)
        meta2 = json.loads((aligned / "metadata.json").read_text(encoding="utf-8"))
        assert meta2["scale_factor_w"] == 2.0, meta2
        assert meta2["scale_factor_h"] == 2.0, meta2
        assert aligner._already_done() is True
        with Image.open(aligned / "images" / "frame_000.png") as img:
            assert img.size == (48, 32), img.size

        stray = aligned / "images" / "old_frame.png"
        _write_image(stray, 8, 8)
        aligner.run(force=True)
        assert not stray.exists(), "stale aligned images should be removed"


if __name__ == "__main__":
    main()
