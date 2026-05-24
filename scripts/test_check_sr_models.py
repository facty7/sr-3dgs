#!/usr/bin/env python3
"""Smoke test SR model preflight CLI."""

import json
import subprocess
import sys


def main():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/check_sr_models.py",
            "--sr_model",
            "real-esrgan",
            "--sr_scale",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    data = json.loads(proc.stdout)
    assert data["weight_name"] == "RealESRGAN_x2plus.pth", data
    assert "download_url" in data, data

    strict = subprocess.run(
        [
            sys.executable,
            "scripts/check_sr_models.py",
            "--sr_model",
            "real-esrgan",
            "--sr_scale",
            "2",
            "--model_path",
            "/missing/realesrgan.pth",
            "--strict",
        ],
        text=True,
        capture_output=True,
    )
    assert strict.returncode == 1, strict.stdout
    missing = json.loads(strict.stdout)
    assert missing["ok"] is False, missing
    assert missing["explicit_exists"] is False, missing


if __name__ == "__main__":
    main()
