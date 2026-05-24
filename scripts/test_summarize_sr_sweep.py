#!/usr/bin/env python3
"""Smoke test SR sweep summarization."""

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _write_min_delivery(out: Path, name: str):
    out.mkdir(parents=True)
    sog = out / f"{name}.sog"
    sog.write_bytes(b"\0" * 32 * 8)
    ply = out / f"{name}_high_quality.ply"
    with ply.open("wb") as f:
        f.write(
            b"ply\nformat binary_little_endian 1.0\n"
            b"element vertex 8\n"
            b"property float x\nproperty float y\nproperty float z\n"
            b"end_header\n"
        )
        for idx in range(8):
            f.write(struct.pack("<fff", float(idx), 0.0, 0.0))
    for filename in ("index.js", "index.css", "settings.json"):
        (out / filename).write_text("", encoding="utf-8")
    (out / "preview.html").write_text(
        f"<html>{name}.sog index.css settings.json</html>",
        encoding="utf-8",
    )
    (out / "START_HERE.html").write_text(f"<html>{name}.sog</html>", encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps({"ok": True, "sog": sog.name, "high_quality_ply": ply.name}),
        encoding="utf-8",
    )
    (out / "diagnostics.json").write_text(
        json.dumps({"ok": True, "files": []}),
        encoding="utf-8",
    )


def _write_min_delivery_manifest_style(out: Path, workspace: Path, name: str):
    out.mkdir(parents=True)
    web_dir = out / "web" / "splat"
    pro_dir = out / "professional"
    web_dir.mkdir(parents=True)
    pro_dir.mkdir()
    splat = web_dir / f"{name}.splat"
    splat.write_bytes(b"\0" * 32 * 8)
    ply = pro_dir / f"{name}_standard.ply"
    with ply.open("wb") as f:
        f.write(
            b"ply\nformat binary_little_endian 1.0\n"
            b"element vertex 8\n"
            b"property float x\nproperty float y\nproperty float z\n"
            b"end_header\n"
        )
        for idx in range(8):
            f.write(struct.pack("<fff", float(idx), 0.0, 0.0))
    (out / "manifest.json").write_text(
        json.dumps({
            "ok": True,
            "files": {
                "splat_file": str(splat),
                "standard_ply": str(ply),
            },
        }),
        encoding="utf-8",
    )
    (workspace / "sr_images").mkdir(parents=True, exist_ok=True)
    (workspace / "sr_images" / "sr_manifest.json").write_text(
        json.dumps({"effective_mode": "resize", "sr_model": "resize", "scale": 2}),
        encoding="utf-8",
    )


def _write_workspace_meta(
    workspace: Path,
    *,
    mode: str,
    selected_count: int,
    target_frames: int,
    psnr: float,
):
    (workspace / "sr_images").mkdir(parents=True)
    (workspace / "reports").mkdir()
    (workspace / "frames").mkdir()
    (workspace / "train_output").mkdir()
    (workspace / "sr_images" / "sr_manifest.json").write_text(
        json.dumps({
            "requested_mode": mode,
            "effective_mode": mode,
            "sr_model": "real-esrgan",
            "scale": 1 if mode == "off" else 2,
            "effective_scale": [1.0, 1.0] if mode == "off" else [2.0, 2.0],
        }),
        encoding="utf-8",
    )
    (workspace / "frames" / "extraction_manifest.json").write_text(
        json.dumps({
            "raw_count": 120,
            "selected_count": selected_count,
            "min_frames": target_frames,
            "selected_pass": "coverage_1" if selected_count >= target_frames else "coverage_3",
            "relaxed": True,
            "projection": "perspective",
        }),
        encoding="utf-8",
    )
    (workspace / "train_output" / "training_summary.json").write_text(
        json.dumps({"best_psnr": psnr, "elapsed_sec": 20.0}),
        encoding="utf-8",
    )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_root = root / "output"
        work_root = root / "workspace"
        output = out_root / "object_off_x1"
        workspace = work_root / output.name
        _write_min_delivery(output, output.name)

        (workspace / "sr_images").mkdir(parents=True)
        (workspace / "reports").mkdir()
        (workspace / "sr_images" / "sr_manifest.json").write_text(
            json.dumps({
                "requested_mode": "model",
                "effective_mode": "off",
                "status": "model_worker_failed_copied_originals",
                "sr_model": "real-esrgan",
                "scale": 1,
                "effective_scale": [1.0, 1.0],
                "output_size": [640, 480],
                "error": "load timeout",
            }),
            encoding="utf-8",
        )
        (workspace / "reports" / "input_quality_frames.json").write_text(
            json.dumps({"verdict": {"score": 80}}),
            encoding="utf-8",
        )
        (workspace / "train_output").mkdir()
        (workspace / "train_output" / "training_summary.json").write_text(
            json.dumps({
                "best_psnr": 24.125,
                "last_loss": 0.0123,
                "elapsed_sec": 12.5,
                "gaussians_final": 1234,
            }),
            encoding="utf-8",
        )
        report = root / "summary.json"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/summarize_sr_sweep.py",
                str(output),
                "--work_root",
                str(work_root),
                "--report",
                str(report),
                "--min_points",
                "1",
            ],
            cwd=str(ROOT),
            check=True,
            text=True,
            capture_output=True,
        )
        assert "object_off_x1" in proc.stdout, proc.stdout
        assert "recommended:" in proc.stdout, proc.stdout
        data = json.loads(report.read_text(encoding="utf-8"))
        row = data["results"][0]
        assert row["sr_mode"] == "off", row
        assert row["sr_requested_mode"] == "model", row
        assert row["sr_scale"] == 1, row
        assert row["sr_effective_scale"] == "1", row
        assert row["sr_fallback"] is True, row
        assert data["analysis"]["fallback_count"] == 1, data
        assert row["input_score"] == 80, row
        assert row["best_psnr"] == 24.125, row
        assert row["train_sec"] == 12.5, row

        delivery = workspace / "delivery"
        _write_min_delivery_manifest_style(delivery, workspace, "object_delivery")
        report2 = root / "summary_delivery.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_sr_sweep.py",
                str(delivery),
                "--work_root",
                str(work_root),
                "--report",
                str(report2),
                "--min_points",
                "1",
            ],
            cwd=str(ROOT),
            check=True,
            text=True,
            capture_output=True,
        )
        row2 = json.loads(report2.read_text(encoding="utf-8"))["results"][0]
        assert row2["sr_mode"] == "resize", row2
        assert row2["point_count"] == 8, row2

        low_cov = out_root / "phone_resize_x2"
        good_cov = out_root / "phone_off_x1"
        _write_min_delivery(low_cov, low_cov.name)
        _write_min_delivery(good_cov, good_cov.name)
        _write_workspace_meta(
            work_root / low_cov.name,
            mode="resize",
            selected_count=18,
            target_frames=64,
            psnr=30.0,
        )
        _write_workspace_meta(
            work_root / good_cov.name,
            mode="off",
            selected_count=72,
            target_frames=64,
            psnr=24.0,
        )
        report3 = root / "summary_coverage.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_sr_sweep.py",
                str(low_cov),
                str(good_cov),
                "--work_root",
                str(work_root),
                "--report",
                str(report3),
                "--min_points",
                "1",
            ],
            cwd=str(ROOT),
            check=True,
            text=True,
            capture_output=True,
        )
        data3 = json.loads(report3.read_text(encoding="utf-8"))
        rows = {Path(row["output"]).name: row for row in data3["results"]}
        assert rows["phone_resize_x2"]["extraction_meets_target"] is False, rows
        assert rows["phone_off_x1"]["extraction_meets_target"] is True, rows
        assert rows["phone_resize_x2"]["extraction_coverage_ratio"] < 1.0, rows
        assert rows["phone_off_x1"]["extraction_coverage_ratio"] >= 1.0, rows
        assert data3["analysis"]["recommended_output"].endswith("phone_off_x1"), data3
        assert data3["analysis"]["low_coverage_count"] == 1, data3
        assert data3["analysis"]["relaxed_extraction_count"] == 2, data3


if __name__ == "__main__":
    main()
