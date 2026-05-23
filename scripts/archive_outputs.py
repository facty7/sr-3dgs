#!/usr/bin/env python3
"""Keep output/ flat by archiving old scene deliveries into workspace_video/."""

import argparse
import html
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _inside(path, parent):
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    return path == parent or parent in path.parents


def _assert_inside(path, parent, label):
    if not _inside(path, parent):
        raise ValueError(f"{label} escapes expected directory: {path}")


def _is_archive_candidate(path, scene):
    name = path.name
    if not path.is_dir() or name == scene:
        return False
    prefixes = (
        f"{scene}_v",
        f"{scene}-v",
        f"{scene}_clean",
        f"{scene}-clean",
        f"{scene}_candidate",
        f"{scene}-candidate",
    )
    return any(name.startswith(prefix) for prefix in prefixes)


def _write_output_index(output_dir):
    scenes = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_dir():
            continue
        start = path / "START_HERE.html"
        manifest = path / "manifest.json"
        if start.exists() and manifest.exists():
            scenes.append(path.name)

    cards = "\n".join(
        f"""    <a class="card" href="{html.escape(scene)}/START_HERE.html">
      <strong>{html.escape(scene)}</strong>
      <span>Open final 3D delivery</span>
    </a>"""
        for scene in scenes
    )
    if not cards:
        cards = "    <p>No published scenes yet.</p>"

    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3D Deliveries</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #11141b; color: #f5f7fb; }}
    main {{ max-width: 920px; margin: 0 auto; padding: 44px 24px; }}
    h1 {{ margin: 0 0 10px; font-size: 34px; }}
    p {{ color: #b9c0ce; line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 28px; }}
    .card {{ display: block; padding: 18px; border: 1px solid #2d3446; border-radius: 8px; background: #181d28; color: #f5f7fb; text-decoration: none; }}
    .card strong {{ display: block; font-size: 18px; margin-bottom: 7px; }}
    .card span {{ color: #aeb8cb; font-size: 14px; }}
    code {{ color: #d2e6ff; }}
  </style>
</head>
<body>
<main>
  <h1>3D Deliveries</h1>
  <p>Only final deliverables live here. Intermediate runs and archived old versions are under <code>workspace_video/&lt;scene&gt;/</code>.</p>
  <div class="grid">
{cards}
  </div>
</main>
</body>
</html>
"""
    (output_dir / "START_HERE.html").write_text(text, encoding="utf-8")


def archive_scene_outputs(scene, output_dir=None, workspace_dir=None, dry_run=False):
    output_dir = Path(output_dir or ROOT / "output")
    workspace_dir = Path(workspace_dir or ROOT / "workspace_video" / scene)
    archive_root = workspace_dir / "archived_outputs"

    _assert_inside(output_dir, ROOT, "output_dir")
    _assert_inside(workspace_dir, ROOT / "workspace_video", "workspace_dir")
    _assert_inside(archive_root, workspace_dir, "archive_root")

    candidates = [path for path in sorted(output_dir.iterdir()) if _is_archive_candidate(path, scene)]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = archive_root / stamp
    moves = []
    for src in candidates:
        dst = archive_dir / src.name
        _assert_inside(src, output_dir, "source")
        _assert_inside(dst, archive_root, "destination")
        moves.append({"from": str(src), "to": str(dst)})
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    if not dry_run:
        _write_output_index(output_dir)
        report = {
            "scene": scene,
            "archived_at": stamp,
            "moves": moves,
            "output_index": str(output_dir / "START_HERE.html"),
        }
        archive_root.mkdir(parents=True, exist_ok=True)
        (archive_root / "latest_archive.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"scene": scene, "dry_run": dry_run, "moves": moves}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="toy")
    parser.add_argument("--output_dir", default=str(ROOT / "output"))
    parser.add_argument("--workspace_dir", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    result = archive_scene_outputs(
        scene=args.scene,
        output_dir=args.output_dir,
        workspace_dir=args.workspace_dir or None,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
