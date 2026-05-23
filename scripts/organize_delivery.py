#!/usr/bin/env python3
"""Organize scattered reconstruction outputs into one delivery folder."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.quality import diagnose_paths, write_delivery_report


def _find_one(root, patterns, required=False):
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"Could not find any of: {patterns} in {root}")
    return None


def organize(package_dir, scene_name=None, delivery_name="delivery_unified"):
    package_dir = Path(package_dir)
    scene_name = scene_name or package_dir.name

    standard_ply = _find_one(
        package_dir,
        ["clean/*_standard.ply", "clean_output/*_standard.ply", "**/clean_opa0.10_standard.ply"],
        required=True,
    )
    splat = _find_one(package_dir, ["*.splat", "web/*.splat", "web/splat/*.splat"])
    viewer = _find_one(package_dir, ["*_viewer.html", "web/*_viewer.html", "web/splat/*_viewer.html"])
    sog = _find_one(package_dir, ["*.sog", "web/sog/*.sog"])
    sog_viewer = _find_one(package_dir, ["*_sog.html", "web/sog/*_sog.html"])
    contact_sheet = _find_one(package_dir, ["preview/contact_sheet.png", "**/contact_sheet.png"])
    turntable = _find_one(package_dir, ["preview/turntable.gif", "**/turntable.gif"])

    targets = [standard_ply]
    if splat:
        targets.append(splat)
    diagnostics = diagnose_paths([str(p) for p in targets])

    results = {"standard_ply": str(standard_ply)}
    if splat:
        results["splat_file"] = str(splat)
    if viewer:
        results["viewer_html"] = str(viewer)
    if sog:
        results["sog_file"] = str(sog)
    if sog_viewer:
        results["sog_viewer_html"] = str(sog_viewer)
    if contact_sheet:
        results["contact_sheet"] = str(contact_sheet)
    if turntable:
        results["turntable"] = str(turntable)

    delivery = write_delivery_report(
        package_dir / delivery_name,
        scene_name,
        results,
        diagnostics,
    )
    manifest = {
        "scene": scene_name,
        "ok": diagnostics["ok"],
        "delivery": str(delivery),
        "primary": {
            "sog_viewer": str((delivery / "web" / "sog" / Path(sog_viewer).name)) if sog_viewer else "",
            "sog": str((delivery / "web" / "sog" / Path(sog).name)) if sog else "",
            "standard_ply": str((delivery / "professional" / Path(standard_ply).name)),
        },
    }
    (delivery / "delivery_index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir")
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--delivery_name", default="delivery_unified")
    args = parser.parse_args()
    manifest = organize(args.package_dir, args.scene_name or None, args.delivery_name)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
