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
        modes = [run["strategy"]["mode"] for run in data["runs"]]
        assert modes == ["off", "resize", "auto", "model"], modes
        for run in data["runs"]:
            cmd = run["command"]
            assert "--sr_mode" in cmd, cmd
            assert "--sr_scale" in cmd, cmd
            assert "--run" not in cmd, cmd


if __name__ == "__main__":
    main()
