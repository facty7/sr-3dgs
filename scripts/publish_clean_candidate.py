#!/usr/bin/env python3
"""Promote a cleaned PLY candidate into a flat output/<scene> delivery.

By default this is lightweight: it stages the candidate PLY and diagnostics
without launching SOG conversion. Pass --convert_sog when you explicitly want
the heavier PlayCanvas conversion step.
"""

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
from sr_3dgs.sog_export import export_sog_viewer


def _inside(path, parent):
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    return path == parent or parent in path.parents


def _assert_inside(path, parent, label):
    if not _inside(path, parent):
        raise ValueError(f"{label} escapes expected directory: {path}")


def _asset_stem(name):
    stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in name).strip("_")
    return stem or "scene"


def _copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _archive_existing(out, scene):
    if not out.exists():
        return ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if _inside(out, ROOT / "output"):
        dst = ROOT / "workspace_video" / scene / "archived_outputs" / stamp / out.name
        _assert_inside(dst, ROOT / "workspace_video" / scene / "archived_outputs", "archive destination")
    elif _inside(out, ROOT / "workspace_video"):
        dst = out.parent / "_archived" / stamp / out.name
        _assert_inside(dst, ROOT / "workspace_video", "archive destination")
    else:
        raise ValueError(f"existing output escapes expected directories: {out}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out), str(dst))
    return str(dst)


def _staging_dir(out):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out.parent / f".{out.name}.staging_{stamp}"


def _patch_preview(html_path, sog_name):
    text = Path(html_path).read_text(encoding="utf-8")
    for old in (Path(html_path).with_suffix(".sog").name, "scene.sog", "toy.sog"):
        text = text.replace(f'fetch("{old}")', f'fetch("{sog_name}")')
        text = text.replace(f"fetch('{old}')", f"fetch('{sog_name}')")
        text = text.replace(f"./{old}", f"./{sog_name}")
    return text


def _write_start_here(out, scene, ply_name, preview_name=None, sog_name=None):
    preview_card = ""
    if preview_name and sog_name:
        preview_card = f"""    <a class="card" href="{preview_name}"><strong>Open 3D Preview</strong><span>PlayCanvas/SOG browser preview.</span></a>
    <a class="card" href="{sog_name}"><strong>Download SOG</strong><span>Compact web/mobile asset.</span></a>"""
    else:
        preview_card = """    <a class="card muted" href="manifest.json"><strong>SOG Not Generated</strong><span>Run with --convert_sog to create the web/mobile asset.</span></a>"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{scene} 3D Delivery</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #11141b; color: #f5f7fb; }}
    main {{ max-width: 900px; margin: 0 auto; padding: 44px 24px; }}
    h1 {{ margin: 0 0 10px; font-size: 34px; }}
    p {{ color: #b9c0ce; line-height: 1.6; }}
    .notice {{ margin: 22px 0 0; padding: 14px 16px; border: 1px solid #394157; border-radius: 8px; background: #171d29; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 28px; }}
    .card {{ display: block; padding: 18px; border: 1px solid #2d3446; border-radius: 8px; background: #181d28; color: #f5f7fb; text-decoration: none; }}
    .card strong {{ display: block; font-size: 18px; margin-bottom: 7px; }}
    .card span {{ color: #aeb8cb; font-size: 14px; }}
    .muted {{ border-style: dashed; }}
    code {{ color: #d2e6ff; }}
  </style>
</head>
<body>
<main>
  <h1>{scene} 3D Delivery</h1>
  <p>This folder contains final deliverables only. Intermediate files stay under <code>workspace_video/{scene}/</code>.</p>
  <div class="notice">
    <strong>Open through local HTTP.</strong>
    <p>Browser security blocks SOG loading from <code>file://</code>. From the project root run <code>python scripts/serve_output.py --scene {scene}</code>, then open the printed URL.</p>
  </div>
  <div class="grid">
{preview_card}
    <a class="card" href="{ply_name}"><strong>Download High Quality PLY</strong><span>SuperSplat / professional tool asset.</span></a>
    <a class="card" href="diagnostics.json"><strong>View Diagnostics</strong><span>Point count, bounds, and quality checks.</span></a>
  </div>
</main>
</body>
</html>
"""
    (out / "START_HERE.html").write_text(html, encoding="utf-8")


def publish_candidate(
    candidate_ply,
    out_dir,
    scene_name,
    asset_name=None,
    convert_sog=False,
    replace=False,
    iterations=None,
    confirm_heavy=False,
):
    candidate_ply = Path(candidate_ply)
    out = Path(out_dir)
    scene = _asset_stem(scene_name)
    asset = _asset_stem(asset_name or scene_name)

    _assert_inside(candidate_ply, ROOT, "candidate_ply")
    if convert_sog:
        if not confirm_heavy:
            raise ValueError(
                "--convert_sog launches PlayCanvas SOG conversion and can be CPU/RAM intensive. "
                "Pass --i_understand_this_is_heavy to confirm."
            )
        _assert_inside(out, ROOT / "output", "out_dir")
    else:
        allowed = _inside(out, ROOT / "workspace_video") or _inside(out, ROOT / "output")
        if not allowed:
            raise ValueError(f"out_dir escapes expected directories: {out}")
    if not candidate_ply.exists():
        raise FileNotFoundError(candidate_ply)
    if out.exists() and not replace:
        raise FileExistsError(f"{out} exists. Pass --replace to archive it first.")

    staging = _staging_dir(out)
    cache_bust = datetime.now().strftime("%Y%m%d%H%M%S")
    _assert_inside(staging, out.parent, "staging directory")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)

    ply_name = f"{asset}_high_quality.ply"
    try:
        _copy(candidate_ply, staging / ply_name)

        preview_name = ""
        sog_name = ""
        if convert_sog:
            viewer_html = staging / "preview.html"
            export_sog_viewer(candidate_ply, viewer_html, overwrite=True, unbundled=True, iterations=iterations)
            produced_sog = viewer_html.with_suffix(".sog")
            if not produced_sog.exists():
                matches = sorted(staging.glob("*.sog"))
                if not matches:
                    raise FileNotFoundError("SOG conversion finished but no .sog file was produced")
                produced_sog = matches[0]
            sog_name = f"{asset}_v{cache_bust}.sog"
            if produced_sog.name != sog_name:
                produced_sog.rename(staging / sog_name)
            preview_name = "preview.html"
            (staging / preview_name).write_text(
                _patch_preview(staging / preview_name, sog_name),
                encoding="utf-8",
            )

        diagnostics = diagnose_paths([str(staging / ply_name)])
        (staging / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        manifest = {
            "scene": scene_name,
            "ok": diagnostics["ok"] and (bool(sog_name) if convert_sog else True),
            "open_first": "START_HERE.html",
            "preview": preview_name,
            "sog": sog_name,
            "high_quality_ply": ply_name,
            "candidate_source": str(candidate_ply),
            "archived_previous_output": "",
            "sog_generated": bool(sog_name),
            "asset_cache_bust": cache_bust,
            "heavy_steps_confirmed": bool(confirm_heavy) if convert_sog else False,
            "notes": [
                "This folder was published from a cleaned PLY candidate.",
                "If sog_generated is false, run this script with --convert_sog before treating it as web/mobile final.",
                "SOG conversion is intentionally gated because it can stress local machines.",
            ],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _write_start_here(staging, scene_name, ply_name, preview_name or None, sog_name or None)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    archived_previous = _archive_existing(out, scene) if out.exists() else ""
    if archived_previous:
        manifest["archived_previous_output"] = archived_previous
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.move(str(staging), str(out))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_ply")
    parser.add_argument(
        "--out",
        required=True,
        help="Use workspace_video/<scene>/review_candidates/... for PLY-only review, or output/<scene> with --convert_sog for final delivery.",
    )
    parser.add_argument("--scene_name", default="toy")
    parser.add_argument("--asset_name", default="")
    parser.add_argument("--convert_sog", action="store_true", help="Run PlayCanvas SOG conversion. This can be heavy.")
    parser.add_argument(
        "--i_understand_this_is_heavy",
        action="store_true",
        help="Required with --convert_sog to confirm that a CPU/RAM-intensive conversion is intended.",
    )
    parser.add_argument("--replace", action="store_true", help="Archive the existing output folder before replacing it.")
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()
    result = publish_candidate(
        args.candidate_ply,
        args.out,
        args.scene_name,
        args.asset_name or None,
        convert_sog=args.convert_sog,
        replace=args.replace,
        iterations=args.iterations,
        confirm_heavy=args.i_understand_this_is_heavy,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
