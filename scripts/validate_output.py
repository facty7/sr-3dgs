#!/usr/bin/env python3
"""Validate a flat output/<scene> delivery folder."""

import argparse
import json
import sys
from pathlib import Path


BASE_REQUIRED = [
    "START_HERE.html",
    "preview.html",
    "diagnostics.json",
    "manifest.json",
    "index.js",
    "index.css",
    "settings.json",
]


def validate(out_dir):
    out = Path(out_dir)
    problems = []
    files = {}
    manifest = {}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("ok", False):
                problems.append("manifest ok is not true")
        except Exception as exc:
            problems.append(f"manifest unreadable: {exc}")

    expected_sog = manifest.get("sog") or _first_name(out, "*.sog")
    expected_ply = manifest.get("high_quality_ply") or _first_name(out, "*_high_quality.ply")
    required = list(BASE_REQUIRED)
    if expected_sog:
        required.append(expected_sog)
    else:
        problems.append("missing SOG asset")
    if expected_ply:
        required.append(expected_ply)
    else:
        problems.append("missing high-quality PLY asset")

    for name in required:
        path = out / name
        if not path.exists():
            problems.append(f"missing {name}")
        else:
            files[name] = path.stat().st_size

    preview_path = out / "preview.html"
    if preview_path.exists():
        text = preview_path.read_text(encoding="utf-8", errors="replace")
        for needle in (expected_sog, "index.css", "settings.json"):
            if needle not in text and needle != "index.js":
                problems.append(f"preview.html does not reference {needle}")

    start_path = out / "START_HERE.html"
    if start_path.exists() and expected_sog:
        text = start_path.read_text(encoding="utf-8", errors="replace")
        if expected_sog not in text:
            problems.append(f"START_HERE.html does not reference {expected_sog}")

    diagnostics_path = out / "diagnostics.json"
    if diagnostics_path.exists():
        try:
            diag = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if not diag.get("ok", False):
                problems.append("diagnostics ok is not true")
        except Exception as exc:
            problems.append(f"diagnostics unreadable: {exc}")

    return {
        "output": str(out),
        "ok": not problems,
        "problems": problems,
        "files": files,
    }


def _first_name(out, pattern):
    matches = sorted(Path(out).glob(pattern))
    return matches[0].name if matches else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = validate(args.output_dir)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
