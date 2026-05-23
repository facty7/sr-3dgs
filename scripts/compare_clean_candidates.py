#!/usr/bin/env python3
"""Compare cleaned PLY candidates against a current delivery PLY."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.quality import diagnose_file


def _manifest_ply(path):
    path = Path(path)
    if path.is_file():
        return path
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = manifest.get("high_quality_ply")
        if name:
            return path / name
    matches = sorted(path.glob("*_high_quality.ply")) or sorted(path.glob("*.ply"))
    if not matches:
        raise FileNotFoundError(f"No PLY found in {path}")
    return matches[0]


def _metrics(path):
    path = _manifest_ply(path)
    diag = diagnose_file(path)
    scale = diag.get("scale_percentiles", {})
    scale_actual = diag.get("scale_actual_percentiles", scale)
    radius = diag.get("radius_percentiles", {})
    opacity = diag.get("opacity_percentiles", {})
    return {
        "path": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "ok": diag["ok"],
        "point_count": diag["count"],
        "radius_p95": radius.get("p95"),
        "radius_p99": radius.get("p99"),
        "radius_max": radius.get("max"),
        "scale_kind": diag.get("scale_kind"),
        "scale_p95": scale.get("p95"),
        "scale_p99": scale.get("p99"),
        "scale_max": scale.get("max"),
        "scale_actual_p95": scale_actual.get("p95"),
        "scale_actual_p99": scale_actual.get("p99"),
        "scale_actual_max": scale_actual.get("max"),
        "opacity_p50": opacity.get("p50"),
        "opacity_p95": opacity.get("p95"),
    }


def _delta(base, item):
    return {
        "points_delta": item["point_count"] - base["point_count"],
        "points_delta_percent": round(
            100.0 * (item["point_count"] - base["point_count"]) / max(1, base["point_count"]),
            2,
        ),
        "size_mb_delta": round(item["size_mb"] - base["size_mb"], 2),
        "radius_p99_delta": _sub(item["radius_p99"], base["radius_p99"]),
        "scale_actual_p99_delta": _sub(item["scale_actual_p99"], base["scale_actual_p99"]),
    }


def _sub(a, b):
    if a is None or b is None:
        return None
    return a - b


def _recommend(base, candidates):
    ranked = []
    for item in candidates:
        d = _delta(base, item)
        retention = item["point_count"] / max(1, base["point_count"])
        radius_penalty = max(0.0, (item["radius_p99"] or 0.0) - (base["radius_p99"] or 0.0))
        scale_penalty = max(0.0, (item["scale_actual_p99"] or 0.0) - (base["scale_actual_p99"] or 0.0))
        score = 100.0
        score -= abs(1.0 - retention) * 60.0
        score -= radius_penalty * 10.0
        score -= scale_penalty * 20.0
        if not item["ok"]:
            score -= 40.0
        ranked.append((score, item, d))
    ranked.sort(key=lambda row: row[0], reverse=True)
    if not ranked:
        return {}
    score, item, d = ranked[0]
    return {
        "path": item["path"],
        "score": round(score, 2),
        "reason": (
            "Highest lightweight candidate score: keeps most points while avoiding larger p99 radius/scale."
        ),
        "delta_from_base": d,
    }


def compare(base, candidates):
    base_metrics = _metrics(base)
    candidate_metrics = [_metrics(path) for path in candidates]
    return {
        "ok": base_metrics["ok"] and all(item["ok"] for item in candidate_metrics),
        "base": base_metrics,
        "candidates": [
            {**item, "delta_from_base": _delta(base_metrics, item)}
            for item in candidate_metrics
        ],
        "recommended_candidate": _recommend(base_metrics, candidate_metrics),
        "notes": [
            "This is a geometry/statistics comparison, not a visual-quality verdict.",
            "Scale scoring uses scale_actual_* values so log-scale PLY files and actual-scale files compare consistently.",
            "Use it to choose which candidate deserves SOG conversion and browser review.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="output/toy")
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    result = compare(args.base, args.candidates)
    text = json.dumps(result, indent=2)
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
