#!/usr/bin/env python3
"""Unit tests for optional super-resolution modes."""

import json
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sr_3dgs.step2_super_resolution import SuperResolutionProcessor
from sr_3dgs.sr_strategy import adjust_strategy_for_model_preflight, recommend_sr_strategy
from sr_3dgs.sr_models.real_esrgan import RealESRGANModel


def _sleep_worker(seconds):
    time.sleep(seconds)


def _write_image(path: Path, width: int = 24, height: int = 16):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    arr[:, :, 1] = 120
    arr[4:12, 6:18, 2] = 240
    Image.fromarray(arr).save(path)


def _read_manifest(output_dir: Path):
    return json.loads((output_dir / "sr_manifest.json").read_text(encoding="utf-8"))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        images = root / "images"
        images.mkdir()
        _write_image(images / "frame_000.png")

        off_dir = root / "off"
        off = SuperResolutionProcessor(
            image_dir=str(images),
            output_dir=str(off_dir),
            sr_model_name="real-esrgan",
            scale=1,
            mode="off",
        )
        off.run(force=True)
        with Image.open(off_dir / "frame_000.png") as img:
            assert img.size == (24, 16), img.size
        off_manifest = _read_manifest(off_dir)
        assert off_manifest["effective_mode"] == "off", off_manifest
        assert off_manifest["model_loaded"] is False, off_manifest
        assert off_manifest["effective_scale"] == [1.0, 1.0], off_manifest

        resize_dir = root / "resize"
        resize = SuperResolutionProcessor(
            image_dir=str(images),
            output_dir=str(resize_dir),
            sr_model_name="real-esrgan",
            scale=2,
            mode="resize",
        )
        resize.run(force=True)
        with Image.open(resize_dir / "frame_000.png") as img:
            assert img.size == (48, 32), img.size
        resize_manifest = _read_manifest(resize_dir)
        assert resize_manifest["effective_mode"] == "resize", resize_manifest
        assert resize_manifest["model_loaded"] is False, resize_manifest
        assert resize_manifest["effective_scale"] == [2.0, 2.0], resize_manifest

        auto_dir = root / "auto_scale1"
        auto = SuperResolutionProcessor(
            image_dir=str(images),
            output_dir=str(auto_dir),
            sr_model_name="real-esrgan",
            scale=1,
            mode="auto",
        )
        auto.run(force=True)
        auto_manifest = _read_manifest(auto_dir)
        assert auto_manifest["effective_mode"] == "off", auto_manifest
        assert auto_manifest["model_loaded"] is False, auto_manifest

        # Reusing one output directory with a different mode must regenerate
        # images instead of silently training on stale resolution.
        shared_dir = root / "shared"
        SuperResolutionProcessor(
            image_dir=str(images),
            output_dir=str(shared_dir),
            sr_model_name="real-esrgan",
            scale=1,
            mode="off",
        ).run(force=True)
        SuperResolutionProcessor(
            image_dir=str(images),
            output_dir=str(shared_dir),
            sr_model_name="real-esrgan",
            scale=2,
            mode="resize",
        ).run(force=False)
        with Image.open(shared_dir / "frame_000.png") as img:
            assert img.size == (48, 32), img.size
        shared_manifest = _read_manifest(shared_dir)
        assert shared_manifest["effective_mode"] == "resize", shared_manifest

        failure_dir = root / "worker_failure"
        SuperResolutionProcessor(
            image_dir=str(images),
            output_dir=str(failure_dir),
            sr_model_name="missing-test-model",
            scale=2,
            mode="model",
            model_load_timeout_s=1,
            frame_timeout_s=1,
        ).run(force=True)
        failure_manifest = _read_manifest(failure_dir)
        assert failure_manifest["effective_mode"] == "off", failure_manifest
        assert failure_manifest["status"] == "model_preflight_failed_copied_originals", failure_manifest
        assert failure_manifest.get("error"), failure_manifest
        assert failure_manifest["model_preflight"]["ok"] is False, failure_manifest
        with Image.open(failure_dir / "frame_000.png") as img:
            assert img.size == (24, 16), img.size

        strict_dir = root / "strict_failure"
        try:
            SuperResolutionProcessor(
                image_dir=str(images),
                output_dir=str(strict_dir),
                sr_model_name="missing-test-model",
                scale=2,
                mode="model",
                model_load_timeout_s=1,
                frame_timeout_s=1,
                strict_model=True,
            ).run(force=True)
            raise AssertionError("strict SR model mode should fail")
        except Exception:
            strict_manifest = _read_manifest(strict_dir)
            assert strict_manifest["effective_mode"] == "model", strict_manifest
            assert strict_manifest["status"] == "model_preflight_failed_strict", strict_manifest

        timeout_wait_dir = root / "timeout_wait"
        timeout_wait_dir.mkdir()
        timeout_processor = SuperResolutionProcessor(
            image_dir=str(images),
            output_dir=str(timeout_wait_dir),
            sr_model_name="real-esrgan",
            scale=2,
            mode="model",
            model_load_timeout_s=1,
            frame_timeout_s=1,
        )
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_sleep_worker, args=(5,))
        proc.start()
        t0 = time.time()
        try:
            timeout_processor._wait_for_worker(proc, [images / "frame_000.png"])
            raise AssertionError("expected SR worker timeout")
        except TimeoutError:
            pass
        assert time.time() - t0 < 4, "SR timeout wait took too long"
        assert not proc.is_alive()

        model_strategy = recommend_sr_strategy({
            "images": {
                "count": 80,
                "dimensions_first_sample": [960, 540],
                "sharpness_laplacian": {"p10": 90.0},
            },
            "verdict": {"score": 78, "problems": []},
        }, preferred_scale=4, vram_gb=24)
        assert model_strategy.mode == "model", model_strategy
        assert model_strategy.scale == 2, model_strategy
        adjusted = adjust_strategy_for_model_preflight(
            model_strategy,
            {"ok": True, "needs_download": True, "weights_exist": False},
            allow_download=False,
        )
        assert adjusted.mode == "resize", adjusted
        assert "weights are not local" in adjusted.reason, adjusted

        allowed = adjust_strategy_for_model_preflight(
            recommend_sr_strategy({
                "images": {
                    "count": 80,
                    "dimensions_first_sample": [960, 540],
                    "sharpness_laplacian": {"p10": 90.0},
                },
                "verdict": {"score": 78, "problems": []},
            }, preferred_scale=2, vram_gb=24),
            {"ok": True, "needs_download": True, "weights_exist": False},
            allow_download=True,
        )
        assert allowed.mode == "model", allowed

        low_coverage_strategy = recommend_sr_strategy({
            "images": {
                "count": 80,
                "dimensions_first_sample": [960, 540],
                "sharpness_laplacian": {"p10": 90.0},
            },
            "verdict": {"score": 78, "problems": []},
            "extraction": {
                "selected_count": 40,
                "min_frames": 64,
                "selected_pass": "coverage_3",
                "passes": [{
                    "name": "coverage_3",
                    "selected_raw_index_coverage": 1.0,
                }],
            },
        }, preferred_scale=2, vram_gb=24)
        assert low_coverage_strategy.mode == "off", low_coverage_strategy
        assert low_coverage_strategy.extraction_coverage_ratio < 1.0, low_coverage_strategy

        low_span_strategy = recommend_sr_strategy({
            "images": {
                "count": 80,
                "dimensions_first_sample": [960, 540],
                "sharpness_laplacian": {"p10": 90.0},
            },
            "verdict": {"score": 78, "problems": []},
            "extraction": {
                "selected_count": 80,
                "min_frames": 64,
                "selected_pass": "coverage_1",
                "passes": [{
                    "name": "coverage_1",
                    "selected_raw_index_coverage": 0.45,
                }],
            },
        }, preferred_scale=2, vram_gb=24)
        assert low_span_strategy.mode == "off", low_span_strategy
        assert low_span_strategy.extraction_temporal_coverage == 0.45, low_span_strategy

        blurry_strategy = recommend_sr_strategy({
            "images": {
                "count": 80,
                "dimensions_first_sample": [960, 540],
                "sharpness_laplacian": {"p10": 20.0},
            },
            "verdict": {"score": 58, "problems": ["blurry_frames"]},
        }, preferred_scale=2, vram_gb=24)
        assert blurry_strategy.mode == "resize", blurry_strategy

        large_strategy = recommend_sr_strategy({
            "images": {
                "count": 120,
                "dimensions_first_sample": [1920, 1080],
                "sharpness_laplacian": {"p10": 120.0},
            },
            "verdict": {"score": 86, "problems": []},
        }, preferred_scale=2, vram_gb=24)
        assert large_strategy.mode == "off", large_strategy

        assert RealESRGANModel(scale=2)._model_scale == 2
        assert RealESRGANModel(scale=4)._model_scale == 4
        assert RealESRGANModel(scale=8)._model_scale == 4
        weights = RealESRGANModel(scale=2).describe_weights()
        assert weights["weight_name"] == "RealESRGAN_x2plus.pth", weights
        assert weights["model_scale"] == 2, weights
        missing_weights = RealESRGANModel(scale=2, model_path="/missing/realesrgan.pth").describe_weights()
        assert missing_weights["ok"] is False, missing_weights
        assert missing_weights["explicit_exists"] is False, missing_weights


if __name__ == "__main__":
    main()
