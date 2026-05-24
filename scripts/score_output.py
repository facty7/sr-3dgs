#!/usr/bin/env python3
"""Score and compare flat output/<scene>, delivery, or workspace outputs.

This is not a perceptual replacement for looking at the scene. It is a
repeatable smoke benchmark for the things that usually break web delivery:
missing assets, excessive file size, invalid bounds, and lost point budget.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_output import validate
from sr_3dgs.quality import diagnose_file


def _resolve_assets(out):
    out = Path(out)
    manifest = _load_manifest(out)
    files = manifest.get("files") or {}

    if files:
        ply = _existing_file(files.get("standard_ply") or files.get("high_quality_ply"))
        sog = _existing_file(files.get("sog_file"))
        splat = _existing_file(files.get("splat_file"))
        return {
            "kind": "delivery",
            "manifest": manifest,
            "validation": {"output": str(out), "ok": bool(manifest.get("ok", False)), "problems": [], "files": {}},
            "sog": sog,
            "splat": splat,
            "ply": ply,
        }

    if (out / "clean_output").exists() or (out / "reports").exists():
        ply = _first_path(out, [
            "clean_output/clean_opa0.10_standard.ply",
            "clean_output/*_standard.ply",
            "delivery/professional/*_standard.ply",
        ])
        sog = _first_path(out, ["*.sog"])
        splat = _first_path(out, ["*.splat", "delivery/web/splat/*.splat"])
        return {
            "kind": "workspace",
            "manifest": manifest,
            "validation": {"output": str(out), "ok": ply is not None, "problems": [] if ply else ["missing PLY file"], "files": {}},
            "sog": sog,
            "splat": splat,
            "ply": ply,
        }

    validation = validate(out)
    sog = out / (manifest.get("sog") or _first_name(out, "*.sog"))
    ply = out / (manifest.get("high_quality_ply") or _first_name(out, "*_high_quality.ply"))
    return {
        "kind": "flat_output",
        "manifest": manifest,
        "validation": validation,
        "sog": sog if sog.exists() else None,
        "splat": None,
        "ply": ply if ply.exists() else None,
    }


def _load_manifest(out):
    path = Path(out) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _existing_file(value):
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def _score_delivery(out, mobile_sog_mb=12.0, min_points=120_000):
    out = Path(out)
    assets = _resolve_assets(out)
    validation = assets["validation"]
    sog = assets["sog"]
    splat = assets["splat"]
    ply = assets["ply"]
    problems = list(validation["problems"])

    if not (sog or splat):
        problems.append("missing web asset")
    if not ply:
        problems.append("missing PLY file")

    ply_diag = diagnose_file(ply) if ply else {}
    web_asset = sog or splat
    sog_mb = web_asset.stat().st_size / (1024 * 1024) if web_asset else 0.0
    ply_mb = ply.stat().st_size / (1024 * 1024) if ply else 0.0
    point_count = int(ply_diag.get("count", 0))

    if sog_mb > mobile_sog_mb:
        problems.append(f"SOG is larger than mobile budget ({sog_mb:.1f}MB > {mobile_sog_mb:.1f}MB)")
    if point_count < min_points:
        problems.append(f"point count is too low ({point_count} < {min_points})")
    if ply_diag and not ply_diag.get("ok", False):
        problems.append("PLY diagnostics failed")

    radius = ply_diag.get("radius_percentiles", {})
    score = 100
    score -= 40 if not validation["ok"] else 0
    score -= 20 if sog_mb > mobile_sog_mb else 0
    score -= 20 if point_count < min_points else 0
    score -= 10 if radius.get("p99", 0) > 20 else 0
    score -= min(10, len(problems) * 3)

    return {
        "output": str(out),
        "kind": assets["kind"],
        "ok": not problems,
        "score": max(0, int(score)),
        "problems": problems,
        "sog_mb": round(sog_mb, 2),
        "ply_mb": round(ply_mb, 2),
        "point_count": point_count,
        "radius_p95": radius.get("p95"),
        "radius_p99": radius.get("p99"),
        "validation": validation,
    }


def _first_name(out, pattern):
    matches = sorted(Path(out).glob(pattern))
    return matches[0].name if matches else ""


def _first_path(out, patterns):
    out = Path(out)
    for pattern in patterns:
        matches = sorted(out.glob(pattern))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="+", help="output/<scene> folders")
    parser.add_argument("--mobile_sog_mb", type=float, default=12.0)
    parser.add_argument("--min_points", type=int, default=120_000)
    args = parser.parse_args()
    results = [
        _score_delivery(path, args.mobile_sog_mb, args.min_points)
        for path in args.outputs
    ]
    print(json.dumps({"ok": all(r["ok"] for r in results), "results": results}, indent=2))
    if not all(r["ok"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
