#!/usr/bin/env python3
"""Recover deliverables from the best available scene artifacts.

This script is intentionally forgiving:
- if a clean standard PLY exists, it uses it
- else if a checkpoint exists, it reruns cleanup
- then it exports web splat, diagnostics, delivery package, and preview
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sr_3dgs.quality import diagnose_paths, write_delivery_report
from sr_3dgs.splat_export import export_from_ply
from sr_3dgs.step5_cleanup import CleanupProcessor
from sr_3dgs.utils import check_dependencies, ensure_dir
from sr_3dgs.web_viewer import generate_viewer


def latest_checkpoint(scene):
    ckpts = sorted(
        (scene / "train_output").glob("checkpoint_step*.pt"),
        key=lambda p: int(p.stem.split("step")[-1]),
    )
    return ckpts[-1] if ckpts else None


def find_clean_dir(scene):
    for name in ["clean_output_v2", "clean_output_fixed", "clean_output"]:
        path = scene / name
        if (path / "clean_opa0.10_standard.ply").exists():
            return path
    return scene / "clean_output"


def run_preview(splat, out_dir):
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "render_splat_preview.py"),
        str(splat),
        "--out_dir",
        str(out_dir),
        "--frames",
        "24",
        "--width",
        "960",
        "--height",
        "720",
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir")
    parser.add_argument("--opacity", type=float, default=0.10)
    parser.add_argument("--preview_opacity", type=float, default=0.50)
    parser.add_argument("--force_cleanup", action="store_true")
    args = parser.parse_args()

    scene = Path(args.scene_dir)
    if not scene.exists():
        raise FileNotFoundError(scene)
    scene_name = scene.name

    deps = check_dependencies()
    report_dir = scene / "reports"
    ensure_dir(report_dir)
    (report_dir / "dependencies.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")

    clean_dir = find_clean_dir(scene)
    standard_ply = clean_dir / f"clean_opa{args.opacity:.2f}_standard.ply"
    web_ply = clean_dir / f"clean_opa{args.opacity:.2f}.ply"

    preview_standard_ply = clean_dir / f"clean_opa{args.preview_opacity:.2f}_standard.ply"
    if (args.force_cleanup or not standard_ply.exists() or not web_ply.exists()
            or not preview_standard_ply.exists()):
        ckpt = latest_checkpoint(scene)
        if not ckpt:
            raise FileNotFoundError("No checkpoint found and no clean standard PLY exists.")
        clean_dir = scene / "clean_output_recovered"
        cleaner = CleanupProcessor(str(ckpt), str(clean_dir), device="cpu")
        cleaner.run(opacity_thresholds=tuple(sorted({args.opacity, args.preview_opacity})))
        standard_ply = clean_dir / f"clean_opa{args.opacity:.2f}_standard.ply"
        web_ply = clean_dir / f"clean_opa{args.opacity:.2f}.ply"
        preview_standard_ply = clean_dir / f"clean_opa{args.preview_opacity:.2f}_standard.ply"

    splat = scene / f"{scene_name}_recovered.splat"
    export_from_ply(str(standard_ply), str(splat), sort_by_depth=True)

    viewer = scene / f"{scene_name}_recovered_viewer.html"
    generate_viewer(str(splat), str(viewer), title=f"{scene_name} recovered", embed_splat=True, max_splat_mb=40)

    diagnostics = diagnose_paths([str(splat), str(standard_ply), str(web_ply)])
    (report_dir / "diagnostics_recovered.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    results = {
        "splat_file": str(splat),
        "viewer_html": str(viewer),
        "standard_ply": str(standard_ply),
    }
    input_manifest = scene / "input_manifest.json"
    if input_manifest.exists():
        results["input_manifest"] = str(input_manifest)
    delivery = write_delivery_report(scene / "delivery_recovered", scene_name, results, diagnostics)

    preview_dir = scene / "preview_recovered"
    preview_splat = scene / f"{scene_name}_preview_opa{args.preview_opacity:.2f}.splat"
    export_from_ply(str(preview_standard_ply), str(preview_splat), sort_by_depth=True)
    run_preview(preview_splat, preview_dir)

    manifest = {
        "scene": scene_name,
        "ok": diagnostics["ok"],
        "delivery": str(delivery),
        "preview": {
            "contact_sheet": str(preview_dir / "contact_sheet.png"),
            "turntable": str(preview_dir / "turntable.gif"),
        },
        "dependencies_ok": deps["ok"],
        "missing_dependencies": deps.get("missing", []),
    }
    (scene / "recovery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = Path(delivery) / "README.txt"
    readme.write_text(
        f"""Recovered 3DGS Delivery: {scene_name}
==============================

Status:
- Diagnostics OK: {diagnostics['ok']}
- Dependency environment OK: {deps['ok']}
- Missing dependencies: {', '.join(deps.get('missing', [])) or 'none'}

Open:
- web/{Path(viewer).name}: embedded HTML viewer smoke test
- web/{Path(splat).name}: lightweight splat asset
- professional/{Path(standard_ply).name}: standard 3DGS PLY for SuperSplat/3DGS-aware tools

Preview:
- {preview_dir / 'contact_sheet.png'}
- {preview_dir / 'turntable.gif'}

Note:
This recovered scene is generated from the existing checkpoint. Install gsplat
and COLMAP/pycolmap to run a full-quality retrain from the source video.
""",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
