#!/usr/bin/env python3
"""Package a trained 3DGS checkpoint into web/pro deliverables."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.quality import diagnose_paths, write_delivery_report
from sr_3dgs.splat_export import export_from_ply
from sr_3dgs.step5_cleanup import CleanupProcessor
from sr_3dgs.utils import ensure_dir
from sr_3dgs.web_viewer import generate_viewer


def _latest_checkpoint(train_dir):
    ckpts = sorted(
        Path(train_dir).glob("checkpoint_step*.pt"),
        key=lambda p: int(p.stem.split("step")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_step*.pt in {train_dir}")
    return ckpts[-1]


def _run_preview(splat, out_dir, width=960, height=720):
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "render_splat_preview.py"),
        str(splat),
        "--out_dir",
        str(out_dir),
        "--frames",
        "24",
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    subprocess.run(cmd, check=True)


def package_scene(scene_name, checkpoint, out_dir, opacity, preview_opacity, viewer_embed):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    clean_dir = out_dir / "clean"
    cleaner = CleanupProcessor(str(checkpoint), str(clean_dir), device="cpu")
    cleaner.run(opacity_thresholds=tuple(sorted({opacity, preview_opacity})))

    standard_ply = clean_dir / f"clean_opa{opacity:.2f}_standard.ply"
    web_ply = clean_dir / f"clean_opa{opacity:.2f}.ply"
    preview_ply = clean_dir / f"clean_opa{preview_opacity:.2f}_standard.ply"

    splat = out_dir / f"{scene_name}.splat"
    export_from_ply(str(standard_ply), str(splat), sort_by_depth=True)
    viewer = out_dir / f"{scene_name}_viewer.html"
    generate_viewer(
        str(splat),
        str(viewer),
        title=scene_name,
        embed_splat=viewer_embed,
        max_splat_mb=40,
    )

    diagnostics = diagnose_paths([str(splat), str(standard_ply), str(web_ply)])
    reports_dir = out_dir / "reports"
    ensure_dir(reports_dir)
    (reports_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    preview_splat = out_dir / f"{scene_name}_preview_opa{preview_opacity:.2f}.splat"
    export_from_ply(str(preview_ply), str(preview_splat), sort_by_depth=True)
    preview_dir = out_dir / "preview"
    _run_preview(preview_splat, preview_dir)

    results = {
        "splat_file": str(splat),
        "viewer_html": str(viewer),
        "standard_ply": str(standard_ply),
    }
    delivery = write_delivery_report(out_dir / "delivery", scene_name, results, diagnostics)
    manifest = {
        "scene": scene_name,
        "checkpoint": str(checkpoint),
        "ok": diagnostics["ok"],
        "splat": str(splat),
        "viewer": str(viewer),
        "standard_ply": str(standard_ply),
        "delivery": str(delivery),
        "preview": {
            "contact_sheet": str(preview_dir / "contact_sheet.png"),
            "turntable": str(preview_dir / "turntable.gif"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--train_dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--opacity", type=float, default=0.10)
    parser.add_argument("--preview_opacity", type=float, default=0.50)
    parser.add_argument("--no_embed", action="store_true")
    args = parser.parse_args()
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    elif args.train_dir:
        checkpoint = _latest_checkpoint(args.train_dir)
    else:
        raise ValueError("Provide --checkpoint or --train_dir")
    manifest = package_scene(
        scene_name=args.scene_name,
        checkpoint=checkpoint,
        out_dir=args.out,
        opacity=args.opacity,
        preview_opacity=args.preview_opacity,
        viewer_embed=not args.no_embed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
