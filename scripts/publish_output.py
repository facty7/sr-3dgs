#!/usr/bin/env python3
"""Publish final deliverables into a flat output/<scene> folder."""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.quality import diagnose_paths


def _copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _find(root, patterns, required=True):
    for pattern in patterns:
        matches = sorted(Path(root).glob(pattern))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find {patterns} in {root}")
    return None


def _asset_stem(scene_name):
    stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in scene_name).strip("_")
    return stem or "scene"


def _write_start_here(out, scene, files):
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{scene} 3D Delivery</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #10131d; color: #f3f5fb; }}
    main {{ max-width: 860px; margin: 0 auto; padding: 48px 24px; }}
    h1 {{ font-size: 34px; margin: 0 0 10px; }}
    p {{ color: #b9c0d4; line-height: 1.6; }}
    .notice {{ margin: 22px 0 0; padding: 14px 16px; border: 1px solid #384156; border-radius: 8px; background: #171d2a; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 28px; }}
    a.card {{ display: block; padding: 18px; border: 1px solid #2b3145; border-radius: 8px; color: #f3f5fb; text-decoration: none; background: #171b28; }}
    a.card strong {{ display: block; font-size: 17px; margin-bottom: 8px; }}
    a.card span {{ color: #aab3cc; font-size: 14px; }}
    code {{ color: #cde3ff; }}
  </style>
</head>
<body>
<main>
  <h1>{scene} 3D Delivery</h1>
  <p>This folder contains final deliverables only. Training, cleanup, and experiment files stay under <code>workspace_video/{scene}</code>.</p>
  <div class="notice">
    <strong>Open through local HTTP.</strong>
    <p>Browser security blocks SOG loading from <code>file://</code>. From the project root run <code>python scripts/serve_output.py --scene {scene}</code>, then open the printed URL.</p>
  </div>
  <div class="grid">
    <a class="card" href="{files['preview']}"><strong>Open 3D Preview</strong><span>PlayCanvas/SOG browser preview.</span></a>
    <a class="card" href="{files['sog']}"><strong>Download SOG</strong><span>Compact web/mobile asset.</span></a>
    <a class="card" href="{files['ply']}"><strong>Download High Quality PLY</strong><span>SuperSplat / professional tool asset.</span></a>
    <a class="card" href="{files['diagnostics']}"><strong>View Diagnostics</strong><span>Point count, bounds, and quality checks.</span></a>
  </div>
</main>
</body>
</html>
"""
    (out / "START_HERE.html").write_text(html, encoding="utf-8")


def _patch_viewer_sog(viewer_text, old_sog_name, new_sog_name):
    for old in (old_sog_name, "toy_fixed_default_crop_sog.sog", "scene.sog", "toy.sog"):
        viewer_text = viewer_text.replace(f'fetch("{old}")', f'fetch("{new_sog_name}")')
        viewer_text = viewer_text.replace(f"fetch('{old}')", f"fetch('{new_sog_name}')")
        viewer_text = viewer_text.replace(f"./{old}", f"./{new_sog_name}")
    return viewer_text


def publish(package_dir, out_dir, scene_name, asset_name=None):
    package_dir = Path(package_dir)
    out = Path(out_dir)
    asset = _asset_stem(asset_name or scene_name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    sog = _find(package_dir, ["*_sog.sog", "*.sog"])
    sog_viewer = _find(package_dir, ["*_sog.html"])
    index_js = _find(package_dir, ["index.js"])
    index_css = _find(package_dir, ["index.css"])
    settings = _find(package_dir, ["settings.json"])
    ply = _find(package_dir, ["clean/*_standard.ply", "**/clean_opa0.10_standard.ply", "*.ply"])
    splat = _find(package_dir, ["*.splat"], required=False)
    contact = _find(package_dir, ["preview/contact_sheet.png", "**/contact_sheet.png"], required=False)
    turntable = _find(package_dir, ["preview/turntable.gif", "**/turntable.gif"], required=False)

    cache_bust = datetime.now().strftime("%Y%m%d%H%M%S")
    sog_name = f"{asset}_v{cache_bust}.sog"
    ply_name = f"{asset}_high_quality.ply"
    splat_name = f"{asset}_legacy.splat"

    _copy(sog, out / sog_name)
    viewer_text = Path(sog_viewer).read_text(encoding="utf-8")
    viewer_text = _patch_viewer_sog(viewer_text, Path(sog).name, sog_name)
    (out / "preview.html").write_text(viewer_text, encoding="utf-8")
    _copy(index_js, out / "index.js")
    _copy(index_css, out / "index.css")
    _copy(settings, out / "settings.json")
    _copy(ply, out / ply_name)
    if splat:
        _copy(splat, out / splat_name)
    if contact:
        _copy(contact, out / "preview_contact_sheet.png")
    if turntable:
        _copy(turntable, out / "preview_turntable.gif")

    diagnostics = diagnose_paths([str(out / ply_name)])
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    manifest = {
        "scene": scene_name,
        "ok": diagnostics["ok"],
        "open_first": "START_HERE.html",
        "preview": "preview.html",
        "sog": sog_name,
        "high_quality_ply": ply_name,
        "workspace_intermediate": str(package_dir),
        "asset_cache_bust": cache_bust,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_start_here(
        out,
        scene_name,
        {
            "preview": "preview.html",
            "sog": sog_name,
            "ply": ply_name,
            "diagnostics": "diagnostics.json",
        },
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene_name", default="toy")
    parser.add_argument("--asset_name", default="", help="Output asset filename stem")
    args = parser.parse_args()
    print(json.dumps(publish(args.package_dir, args.out, args.scene_name, args.asset_name or None), indent=2))


if __name__ == "__main__":
    main()
