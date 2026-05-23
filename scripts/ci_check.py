#!/usr/bin/env python3
"""CI-safe checks that do not require local output assets or browsers."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(name, cmd):
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "name": name,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_output", action="store_true",
                        help="Also run output/toy checks when local assets exist")
    args = parser.parse_args()
    py = sys.executable
    checks = [
        ("syntax", [py, "scripts/check_syntax.py", "scripts", "sr_3dgs"]),
        ("cluster_clean_unit", [py, "scripts/test_cluster_clean.py"]),
        ("candidate_compare_unit", [py, "scripts/test_candidate_compare.py"]),
        ("scene_input_unit", [py, "scripts/test_assess_scene_inputs.py"]),
        ("video_pipeline_input_quality_unit", [py, "scripts/test_video_pipeline_input_quality.py"]),
        ("core_crop_unit", [py, "scripts/test_crop_ply_by_core.py"]),
        ("confidence_filter_unit", [py, "scripts/test_filter_ply_confidence.py"]),
        ("publish_candidate_unit", [py, "scripts/test_publish_clean_candidate.py"]),
        ("publish_output_unit", [py, "scripts/test_publish_output.py"]),
        ("repo_audit", [py, "scripts/audit_repo.py"]),
    ]
    if args.include_output:
        checks.extend([
            ("validate_output", [py, "scripts/validate_output.py", "output/toy"]),
            ("score_output", [py, "scripts/score_output.py", "output/toy"]),
            ("http_preview_smoke", [py, "scripts/http_preview_smoke.py", "output/toy"]),
        ])
    results = [_run(name, cmd) for name, cmd in checks]
    report = {"ok": all(item["ok"] for item in results), "checks": results}
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
