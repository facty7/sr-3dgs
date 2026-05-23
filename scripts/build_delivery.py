#!/usr/bin/env python3
"""Build a small delivery folder from a processed scene."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def copy_if_exists(src, dst):
    src = Path(src)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def run_capture(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", help="Processed scene directory, e.g. workspace_video/toy")
    parser.add_argument("--out", default="", help="Delivery folder")
    args = parser.parse_args()

    scene = Path(args.scene_dir)
    if not scene.exists():
        raise FileNotFoundError(scene)

    name = scene.name
    out = Path(args.out) if args.out else scene / "delivery"
    out.mkdir(parents=True, exist_ok=True)

    splat = scene / f"{name}.splat"
    fixed_splat = scene / f"{name}_fixed.splat"
    if fixed_splat.exists():
        splat = fixed_splat

    viewer = scene / f"{name}_viewer.html"
    fixed_viewer = scene / f"{name}_fixed_viewer.html"
    if fixed_viewer.exists():
        viewer = fixed_viewer

    clean_dir = scene / "clean_output"
    for candidate in ["clean_output_v2", "clean_output_fixed", "clean_output"]:
        if (scene / candidate).exists():
            clean_dir = scene / candidate
            break

    standard_ply = clean_dir / "clean_opa0.10_standard.ply"
    web_ply = clean_dir / "clean_opa0.10.ply"

    copied = {}
    copied["web_splat"] = copy_if_exists(splat, out / "web" / splat.name)
    copied["viewer_html"] = copy_if_exists(viewer, out / "web" / viewer.name)
    copied["standard_ply"] = copy_if_exists(standard_ply, out / "professional" / standard_ply.name)
    copied["web_ply"] = copy_if_exists(web_ply, out / "professional" / web_ply.name)
    copy_if_exists(scene / "input_manifest.json", out / "reports" / "input_manifest.json")

    diag_targets = [p for p in [splat, standard_ply, web_ply] if p.exists()]
    diagnostics = ""
    if diag_targets:
        diagnostics = run_capture([sys.executable, "scripts/diagnose_scene.py", *map(str, diag_targets)])
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "diagnostics.txt").write_text(diagnostics, encoding="utf-8")

    manifest = {
        "scene": name,
        "source_dir": str(scene),
        "files": copied,
        "notes": [
            "web/*.splat is for lightweight browser pipelines.",
            "professional/*_standard.ply keeps standard 3DGS log-scales and opacity logits for tools such as SuperSplat.",
            "reports/diagnostics.txt should say OK for coordinate range before delivery.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)
    if diagnostics:
        print(diagnostics)


if __name__ == "__main__":
    main()
