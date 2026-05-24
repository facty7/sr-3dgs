#!/usr/bin/env python3
"""Summarize SR sweep results across output/workspace folders."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_output import _score_delivery


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _outputs_from_plan(plan_path):
    plan_path = Path(plan_path)
    plan = _read_json(plan_path)
    final_root = plan.get("final_output_root")
    outputs = []
    skipped = []
    for run in plan.get("runs") or []:
        output_dir = run.get("output_dir")
        if not output_dir and final_root and run.get("output_name"):
            output_dir = str(Path(final_root) / run["output_name"])
        if not output_dir:
            skipped.append({
                "output_name": run.get("output_name", ""),
                "reason": "missing output_dir",
            })
            continue
        output_path = Path(output_dir)
        if output_path.exists():
            outputs.append(str(output_path))
        else:
            skipped.append({
                "output_name": run.get("output_name", output_path.name),
                "output_dir": str(output_path),
                "reason": "output directory not found",
            })
    return plan, outputs, skipped


def _workspace_for_output(output_dir, work_root):
    output_dir = Path(output_dir)
    if (output_dir / "sr_images" / "sr_manifest.json").exists():
        return output_dir
    if output_dir.name == "delivery":
        parent = output_dir.parent
        if (parent / "sr_images" / "sr_manifest.json").exists():
            return parent
    if not work_root:
        return None
    candidate = Path(work_root) / output_dir.name
    return candidate if candidate.exists() else None


def _summarize(output_dir, work_root, mobile_sog_mb, min_points):
    output_dir = Path(output_dir)
    workspace = _workspace_for_output(output_dir, work_root)
    score = _score_delivery(output_dir, mobile_sog_mb, min_points)

    sr_manifest = {}
    sr_strategy = {}
    input_quality = {}
    colmap_report = {}
    training_summary = {}
    extraction_manifest = {}
    if workspace:
        sr_manifest = _read_json(workspace / "sr_images" / "sr_manifest.json")
        sr_strategy = _read_json(workspace / "reports" / "sr_strategy.json")
        input_quality = _read_json(workspace / "reports" / "input_quality_frames.json")
        colmap_report = _read_json(workspace / "colmap" / "colmap_report.json")
        training_summary = _read_json(workspace / "train_output" / "training_summary.json")
        extraction_manifest = _read_json(workspace / "frames" / "extraction_manifest.json")
    if not extraction_manifest:
        extraction_manifest = _read_json(output_dir / "reports" / "extraction_manifest.json")
    if not colmap_report:
        colmap_report = _read_json(output_dir / "reports" / "colmap_report.json")

    selected_frames = extraction_manifest.get("selected_count")
    target_frames = extraction_manifest.get("min_frames")
    coverage_ratio = _coverage_ratio(selected_frames, target_frames)
    coverage_meets_target = None if coverage_ratio is None else coverage_ratio >= 1.0
    selected_pass = _selected_extraction_pass(extraction_manifest)
    temporal_target = _temporal_target(extraction_manifest)

    return {
        "output": str(output_dir),
        "workspace": str(workspace) if workspace else "",
        "score": score["score"],
        "ok": score["ok"],
        "problems": score["problems"],
        "sr_mode": sr_manifest.get("effective_mode") or sr_strategy.get("mode") or "",
        "sr_requested_mode": sr_manifest.get("requested_mode") or sr_strategy.get("mode") or "",
        "sr_model": sr_manifest.get("sr_model") or sr_strategy.get("model") or "",
        "sr_scale": sr_manifest.get("scale") or sr_strategy.get("scale") or "",
        "sr_effective_scale": _scale_label(sr_manifest.get("effective_scale")),
        "sr_status": sr_manifest.get("status") or "",
        "sr_fallback": _is_fallback(sr_manifest),
        "sr_error": sr_manifest.get("error") or "",
        "sr_needs_download": (sr_manifest.get("model_preflight") or {}).get("needs_download"),
        "sr_weights_exist": (sr_manifest.get("model_preflight") or {}).get("weights_exist"),
        "sr_strategy_reason": sr_strategy.get("reason") or "",
        "output_size": sr_manifest.get("output_size") or [],
        "extraction_selected_count": selected_frames,
        "extraction_raw_count": extraction_manifest.get("raw_count"),
        "extraction_target_frames": target_frames,
        "extraction_coverage_ratio": coverage_ratio,
        "extraction_meets_target": coverage_meets_target,
        "extraction_selected_pass": extraction_manifest.get("selected_pass") or "",
        "extraction_relaxed": extraction_manifest.get("relaxed"),
        "extraction_projection": extraction_manifest.get("projection") or "",
        "extraction_temporal_coverage": selected_pass.get("selected_raw_index_coverage"),
        "extraction_temporal_target": temporal_target,
        "extraction_temporal_thinned_count": selected_pass.get("temporal_thinned_count"),
        "colmap_ok": colmap_report.get("ok"),
        "colmap_camera_model": colmap_report.get("selected_camera_model") or colmap_report.get("camera_model") or "",
        "colmap_registered_images": colmap_report.get("registered_images"),
        "colmap_image_count": colmap_report.get("image_count"),
        "colmap_registered_ratio": colmap_report.get("registered_ratio"),
        "colmap_points3d": colmap_report.get("points3d"),
        "colmap_meets_quality_target": colmap_report.get("meets_quality_target"),
        "colmap_attempt_count": len(colmap_report.get("attempts") or []),
        "input_score": (input_quality.get("verdict") or {}).get("score"),
        "sog_mb": score["sog_mb"],
        "ply_mb": score["ply_mb"],
        "point_count": score["point_count"],
        "radius_p99": score["radius_p99"],
        "best_psnr": training_summary.get("best_psnr"),
        "last_loss": training_summary.get("last_loss"),
        "train_sec": training_summary.get("elapsed_sec"),
        "gaussians_final": training_summary.get("gaussians_final"),
    }


def _print_table(rows):
    if not rows:
        return
    headers = [
        "score", "mode", "req", "model", "scale", "eff_x", "fb", "frames",
        "target", "cov", "span", "cam", "cam_ratio", "pass", "psnr", "train_s",
        "points", "sog_mb", "ply_mb", "output"
    ]
    widths = {h: len(h) for h in headers}
    values = []
    for row in rows:
        item = {
            "score": str(row["score"]),
            "mode": str(row["sr_mode"]),
            "req": str(row.get("sr_requested_mode") or "-"),
            "model": str(row["sr_model"]),
            "scale": str(row["sr_scale"]),
            "eff_x": str(row.get("sr_effective_scale") or "-"),
            "fb": "yes" if row.get("sr_fallback") else "-",
            "frames": _fmt_optional(row.get("extraction_selected_count")),
            "target": _fmt_optional(row.get("extraction_target_frames")),
            "cov": _fmt_ratio(row.get("extraction_coverage_ratio")),
            "span": _fmt_ratio(row.get("extraction_temporal_coverage")),
            "cam": str(row.get("colmap_camera_model") or "-"),
            "cam_ratio": _fmt_ratio(row.get("colmap_registered_ratio")),
            "pass": str(row.get("extraction_selected_pass") or "-"),
            "psnr": _fmt_optional(row.get("best_psnr")),
            "train_s": _fmt_optional(row.get("train_sec")),
            "points": str(row["point_count"]),
            "sog_mb": str(row["sog_mb"]),
            "ply_mb": str(row["ply_mb"]),
            "output": Path(row["output"]).name,
        }
        values.append(item)
        for key, value in item.items():
            widths[key] = max(widths[key], len(value))

    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for item in values:
        print("  ".join(item[h].ljust(widths[h]) for h in headers))


def _is_fallback(sr_manifest):
    if not sr_manifest:
        return False
    requested = sr_manifest.get("requested_mode")
    effective = sr_manifest.get("effective_mode")
    status = sr_manifest.get("status") or ""
    return (
        bool(requested and effective and requested != effective)
        or "fallback" in status
        or "failed_copied_originals" in status
    )


def _analysis(rows):
    if not rows:
        return {}
    ok_rows = [row for row in rows if row.get("ok")]
    scored = ok_rows or rows
    winner = max(
        scored,
        key=lambda row: (
            _coverage_rank(row),
            _temporal_rank(row),
            _colmap_rank(row),
            row.get("score", 0),
            row.get("best_psnr") if row.get("best_psnr") is not None else -1,
            row.get("point_count", 0),
            -float(row.get("train_sec") or 0),
        ),
    )
    fallback_rows = [row for row in rows if row.get("sr_fallback")]
    resize_rows = [row for row in rows if row.get("sr_mode") == "resize"]
    low_coverage_rows = [
        row for row in rows
        if row.get("extraction_meets_target") is False
    ]
    low_temporal_rows = [
        row for row in rows
        if _has_low_temporal_coverage(row)
    ]
    low_colmap_rows = [
        row for row in rows
        if _has_low_colmap_quality(row)
    ]
    relaxed_rows = [row for row in rows if row.get("extraction_relaxed") is True]
    missing_coverage_rows = [
        row for row in rows
        if row.get("extraction_meets_target") is None
    ]
    notes = []
    if fallback_rows:
        notes.append(
            f"{len(fallback_rows)} run(s) fell back; inspect sr_error/model_preflight before trusting learned SR."
        )
    if low_coverage_rows:
        notes.append(
            f"{len(low_coverage_rows)} run(s) missed the extraction coverage target; prefer higher-coverage runs before judging SR."
        )
    if low_temporal_rows:
        notes.append(
            f"{len(low_temporal_rows)} run(s) cover only part of the source timeline; prefer full-turn coverage for phone videos."
        )
    if low_colmap_rows:
        notes.append(
            f"{len(low_colmap_rows)} run(s) have weak COLMAP registration; review colmap_report.json before trusting training metrics."
        )
    if relaxed_rows:
        notes.append(
            f"{len(relaxed_rows)} run(s) used relaxed extraction thresholds; inspect the frame contact sheet for blur."
        )
    if missing_coverage_rows:
        notes.append(
            f"{len(missing_coverage_rows)} run(s) lack extraction coverage metadata; rerun extraction with current code for stronger comparison."
        )
    if resize_rows:
        best_resize = max(resize_rows, key=lambda row: row.get("point_count", 0))
        off_like = [row for row in rows if row.get("sr_effective_scale") == "1"]
        if off_like and best_resize.get("point_count", 0) < max(r.get("point_count", 0) for r in off_like):
            notes.append("resize did not improve point count on this sweep; prefer off/auto unless visual review says otherwise.")
    reason = "highest score among runs that meet extraction coverage, with PSNR/point-count tie-breakers; still requires visual review"
    if winner.get("extraction_meets_target") is False:
        reason = "best available score, but extraction coverage target was missed; capture/rerun coverage should be reviewed first"
    elif _has_low_temporal_coverage(winner):
        reason = "best available score, but selected frames cover only part of the source timeline; review capture coverage first"
    elif _has_low_colmap_quality(winner):
        reason = "best available score, but COLMAP registered too few images; review camera reconstruction first"
    return {
        "recommended_output": winner["output"],
        "recommended_reason": reason,
        "fallback_count": len(fallback_rows),
        "low_coverage_count": len(low_coverage_rows),
        "low_temporal_coverage_count": len(low_temporal_rows),
        "low_colmap_count": len(low_colmap_rows),
        "relaxed_extraction_count": len(relaxed_rows),
        "notes": notes,
    }


def _fmt_optional(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _fmt_ratio(value):
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def _coverage_ratio(selected, target):
    try:
        selected = float(selected)
        target = float(target)
    except (TypeError, ValueError):
        return None
    if target <= 0:
        return None
    return round(selected / target, 3)


def _coverage_rank(row):
    meets = row.get("extraction_meets_target")
    if meets is True:
        return 2
    if meets is None:
        return 1
    return 0


def _temporal_rank(row):
    value = row.get("extraction_temporal_coverage")
    target = row.get("extraction_temporal_target") or 0.80
    if value is None:
        return 1
    try:
        return 2 if float(value) >= float(target) else 0
    except (TypeError, ValueError):
        return 1


def _has_low_temporal_coverage(row):
    value = row.get("extraction_temporal_coverage")
    target = row.get("extraction_temporal_target") or 0.80
    try:
        return value is not None and float(value) < float(target)
    except (TypeError, ValueError):
        return False


def _colmap_rank(row):
    meets = row.get("colmap_meets_quality_target")
    if meets is True:
        return 2
    if meets is False:
        return 0
    ratio = row.get("colmap_registered_ratio")
    if ratio is None:
        return 1
    try:
        return 2 if float(ratio) >= 0.45 else 0
    except (TypeError, ValueError):
        return 1


def _has_low_colmap_quality(row):
    meets = row.get("colmap_meets_quality_target")
    if meets is False:
        return True
    ratio = row.get("colmap_registered_ratio")
    try:
        return ratio is not None and float(ratio) < 0.45
    except (TypeError, ValueError):
        return False


def _temporal_target(extraction_manifest):
    try:
        return float(extraction_manifest.get("min_span"))
    except (TypeError, ValueError):
        return 0.80


def _selected_extraction_pass(extraction_manifest):
    selected = extraction_manifest.get("selected_pass")
    for item in extraction_manifest.get("passes") or []:
        if item.get("name") == selected:
            return item
    return {}


def _scale_label(value):
    if not value:
        return ""
    try:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            sx = float(value[0])
            sy = float(value[1])
            if abs(sx - sy) < 0.01:
                return f"{sx:.2g}"
            return f"{sx:.2g}x{sy:.2g}"
        return f"{float(value):.2g}"
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="*", help="output folders from a sweep")
    parser.add_argument("--plan", default="",
                        help="SR sweep plan JSON from scripts/plan_sr_sweep.py")
    parser.add_argument("--work_root", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--mobile_sog_mb", type=float, default=12.0)
    parser.add_argument("--min_points", type=int, default=120_000)
    args = parser.parse_args()

    plan = {}
    skipped_outputs = []
    outputs = list(args.outputs)
    work_root = args.work_root
    if args.plan:
        plan, plan_outputs, skipped_outputs = _outputs_from_plan(args.plan)
        outputs.extend(plan_outputs)
        if not work_root:
            work_root = plan.get("work_root", "")
    if not work_root:
        work_root = "workspace_video/sr_sweeps"
    if not outputs:
        parser.error("provide output folders or --plan with completed output directories")

    rows = [
        _summarize(path, work_root, args.mobile_sog_mb, args.min_points)
        for path in outputs
    ]
    rows.sort(
        key=lambda row: (
            _coverage_rank(row),
            _temporal_rank(row),
            _colmap_rank(row),
            row["score"],
            row["point_count"],
        ),
        reverse=True,
    )
    report = {
        "ok": all(row["ok"] for row in rows),
        "plan": str(Path(args.plan)) if args.plan else "",
        "skipped_outputs": skipped_outputs,
        "analysis": _analysis(rows),
        "results": rows,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_table(rows)
    analysis = report["analysis"]
    if analysis:
        print(
            "\nrecommended: "
            f"{Path(analysis['recommended_output']).name} "
            f"({analysis['recommended_reason']})"
        )
        for note in analysis.get("notes", []):
            print(f"note: {note}")
    for item in skipped_outputs:
        print(
            "skipped: "
            f"{item.get('output_name') or item.get('output_dir')} "
            f"({item.get('reason')})"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
