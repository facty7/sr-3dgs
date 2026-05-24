#!/usr/bin/env python3
"""Smoke test video pipeline input-quality report wiring."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from sr_3dgs.video_pipeline import VideoPipeline, VideoPipelineConfig
from sr_3dgs.quality import write_delivery_report


def _write_frame(path, idx):
    arr = np.full((56, 72, 3), 36 + idx * 18, dtype=np.uint8)
    arr[12:38, 18 + idx:48 + idx] = 240 - idx * 8
    Image.fromarray(arr).save(path)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = VideoPipelineConfig(
            video_path=str(tmp_path / "phone_clip.mp4"),
            work_dir=str(tmp_path / "workspace"),
            output_name="phone_clip",
            assess_inputs=True,
        )
        pipeline = VideoPipeline(cfg)
        pipeline.frames_dir.mkdir(parents=True)
        for idx in range(6):
            _write_frame(pipeline.frames_dir / f"frame_{idx:03d}.png", idx)
        extraction_manifest = pipeline.frames_dir / "extraction_manifest.json"
        extraction_manifest.write_text('{"selected_pass":"coverage_1"}\n', encoding="utf-8")

        results = {}
        results["extraction_manifest"] = str(extraction_manifest)
        pipeline._assess_inputs(
            results,
            image_dir=pipeline.frames_dir,
            report_name="input_quality_frames",
        )

        report_json = Path(results["input_quality_frames"])
        report_html = report_json.with_suffix(".html")
        assert report_json.exists(), results
        assert report_html.exists(), results

        delivery = tmp_path / "delivery"
        write_delivery_report(
            delivery,
            "phone_clip",
            results,
            {"ok": True, "files": []},
        )
        manifest = (delivery / "manifest.json").read_text(encoding="utf-8")
        assert "input_quality_frames" in manifest, manifest
        assert "extraction_manifest" in manifest, manifest
        assert (delivery / "reports" / "input_quality_frames.json").exists()
        assert (delivery / "reports" / "input_quality_frames.html").exists()
        assert (delivery / "reports" / "extraction_manifest.json").exists()


if __name__ == "__main__":
    main()
