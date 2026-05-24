#!/usr/bin/env python3
"""Unit tests for COLMAP fallback selection without running COLMAP."""

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from sr_3dgs.step1_colmap import COLMAPExtractor


class FakeCOLMAPExtractor(COLMAPExtractor):
    def __init__(self, *args, stats_by_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats_by_model = stats_by_model
        self.current_stats = (0, 0)

    def _run_colmap_once(self):
        self.current_stats = self.stats_by_model[self.camera_model]
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (self.sparse_dir / name).write_bytes(b"ok")

    def _read_reconstruction_stats(self):
        return self.current_stats


def _write_image(path, idx):
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :] = 20 + idx
    arr[8:24, 8:24] = 220
    Image.fromarray(arr).save(path)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        images = root / "images"
        images.mkdir()
        for idx in range(20):
            _write_image(images / f"frame_{idx:03d}.png", idx)

        extractor = FakeCOLMAPExtractor(
            image_dir=str(images),
            work_dir=str(root / "colmap"),
            camera_model="SIMPLE_PINHOLE",
            camera_model_candidates=("SIMPLE_RADIAL", "PINHOLE"),
            min_registered_ratio=0.75,
            min_registered_images=12,
            stats_by_model={
                "SIMPLE_PINHOLE": (8, 120),
                "SIMPLE_RADIAL": (17, 900),
                "PINHOLE": (15, 700),
            },
        )
        sparse = extractor.run(force=True)
        assert sparse == extractor.sparse_dir, sparse

        report = json.loads(extractor.report_path.read_text(encoding="utf-8"))
        assert report["selected_camera_model"] == "SIMPLE_RADIAL", report
        assert report["registered_images"] == 17, report
        assert report["registered_ratio"] == 0.85, report
        assert report["meets_quality_target"] is True, report
        assert len(report["attempts"]) == 2, report
        assert report["attempts"][0]["meets_quality_target"] is False, report
        assert report["attempts"][1]["meets_quality_target"] is True, report

        dedupe = FakeCOLMAPExtractor(
            image_dir=str(images),
            work_dir=str(root / "colmap_dedupe"),
            camera_model="simple_pinhole",
            camera_model_candidates=("SIMPLE_RADIAL", "simple_pinhole", "PINHOLE"),
            stats_by_model={},
        )
        sequence = dedupe._camera_model_sequence()
        assert sequence == ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "PINHOLE"], sequence


if __name__ == "__main__":
    main()
