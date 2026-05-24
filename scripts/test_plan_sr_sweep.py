#!/usr/bin/env python3
"""Smoke test SR sweep plan generation."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    with tempfile.TemporaryDirectory() as tmp:
        plan = Path(tmp) / "plan.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/plan_sr_sweep.py",
                "--video",
                "input_videos/object.mp4",
                "--preset",
                "debug",
                "--no_showcase",
                "--plan",
                str(plan),
            ],
            cwd=str(ROOT),
            check=True,
            text=True,
            capture_output=True,
        )
        data = json.loads(plan.read_text(encoding="utf-8"))
        assert len(data["runs"]) == 4, data
        assert data["work_root"].endswith("sr_sweeps"), data
        assert data["final_output_root"].endswith("sr_sweeps"), data
        assert data["summary_command"][1] == "scripts/summarize_sr_sweep.py", data
        assert "--plan" in data["summary_command"], data
        modes = [run["strategy"]["mode"] for run in data["runs"]]
        assert modes == ["off", "resize", "auto", "model"], modes
        for run in data["runs"]:
            cmd = run["command"]
            assert "--sr_mode" in cmd, cmd
            assert "--sr_scale" in cmd, cmd
            assert "--run" not in cmd, cmd
            assert run["workspace_dir"].endswith(run["output_name"]), run
            assert run["output_dir"].endswith(run["output_name"]), run

        phone_plan = Path(tmp) / "phone_plan.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/plan_sr_sweep.py",
                "--video",
                "input_videos/object.mp4",
                "--preset",
                "debug",
                "--phone_coverage_sweep",
                "--strategy",
                "off:1",
                "--strategy",
                "resize:2",
                "--extract_variant",
                "dense:120:360:4:span0.9:adaptive",
                "--no_showcase",
                "--plan",
                str(phone_plan),
            ],
            cwd=str(ROOT),
            check=True,
            text=True,
            capture_output=True,
        )
        phone_data = json.loads(phone_plan.read_text(encoding="utf-8"))
        assert phone_data["phone_coverage_sweep"] is True, phone_data
        assert phone_data["summary_report"].endswith("sr_sweep_summary.json"), phone_data
        assert len(phone_data["runs"]) == 8, phone_data
        variants = [run["extraction_variant"].get("name") for run in phone_data["runs"]]
        assert variants.count("cover64") == 2, variants
        assert variants.count("cover96") == 2, variants
        assert variants.count("strict64") == 2, variants
        assert variants.count("dense") == 2, variants
        dense_run = next(
            run for run in phone_data["runs"]
            if run["extraction_variant"].get("name") == "dense"
        )
        dense_cmd = dense_run["command"]
        assert dense_run["output_name"].startswith("object_dense_"), dense_run
        assert "--extract_min_frames" in dense_cmd and "120" in dense_cmd, dense_cmd
        assert "--extract_max_frames" in dense_cmd and "360" in dense_cmd, dense_cmd
        assert "--extract_fps" in dense_cmd and "4.0" in dense_cmd, dense_cmd
        assert "--extract_min_span" in dense_cmd and "0.9" in dense_cmd, dense_cmd
        strict_run = next(
            run for run in phone_data["runs"]
            if run["extraction_variant"].get("name") == "strict64"
        )
        assert "--no_adaptive_extract" in strict_run["command"], strict_run
        assert "--extract_min_span" in strict_run["command"], strict_run


if __name__ == "__main__":
    main()
