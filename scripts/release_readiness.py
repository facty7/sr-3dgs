#!/usr/bin/env python3
"""Summarize whether the repo is ready to publish or release locally."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


REQUIRED_SOURCE_FILES = [
    "README.md",
    "README.zh-CN.md",
    "README.ja.md",
    "USAGE.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "requirements-optional.txt",
    "setup.py",
    ".gitignore",
    "CONTRIBUTING.md",
    ".github/workflows/ci.yml",
    "docs/API.md",
    "docs/AUTODL.md",
    "docs/CAPTURE_GUIDE.md",
    "docs/CLOUD.md",
    "docs/PROJECT_POSITIONING.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/ROADMAP_PIPELINE.md",
    "docs/assets/toy-preview-cleaned.png",
    "docs/assets/toy-contact-sheet.png",
    "scripts/ci_check.py",
    "scripts/smoke_check.py",
    "scripts/benchmark_outputs.py",
    "scripts/test_cluster_clean.py",
    "scripts/test_candidate_compare.py",
    "scripts/test_assess_scene_inputs.py",
    "scripts/test_video_extractor_adaptive.py",
    "scripts/test_colmap_fallbacks.py",
    "scripts/test_super_resolution_modes.py",
    "scripts/test_check_sr_models.py",
    "scripts/test_video_pipeline_input_quality.py",
    "scripts/test_intrinsic_alignment_cache.py",
    "scripts/test_plan_sr_sweep.py",
    "scripts/test_summarize_sr_sweep.py",
    "scripts/test_crop_ply_by_core.py",
    "scripts/test_filter_ply_confidence.py",
    "scripts/test_publish_clean_candidate.py",
    "scripts/test_publish_output.py",
    "scripts/cluster_clean_ply.py",
    "scripts/crop_ply_by_core.py",
    "scripts/filter_ply_confidence.py",
    "scripts/compare_clean_candidates.py",
    "scripts/assess_scene_inputs.py",
    "scripts/check_sr_models.py",
    "scripts/plan_sr_sweep.py",
    "scripts/summarize_sr_sweep.py",
    "scripts/preflight_heavy.py",
    "scripts/render_ply_contact_sheet.py",
    "scripts/build_candidate_review.py",
    "scripts/archive_outputs.py",
    "scripts/publish_clean_candidate.py",
    "scripts/publish_output.py",
    "scripts/serve_output.py",
    "scripts/http_preview_smoke.py",
    "configs/benchmark_outputs.json",
    "sr_3dgs/sr_strategy.py",
    "sr_3dgs/py.typed",
]


def _run(cmd):
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _exists_report(paths):
    return {path: (ROOT / path).exists() for path in paths}


def release_readiness(include_output=False):
    py = sys.executable
    source_files = _exists_report(REQUIRED_SOURCE_FILES)
    missing = [path for path, exists in source_files.items() if not exists]
    ci = _run([py, "scripts/ci_check.py"])
    output_check = None
    if include_output:
        output_check = _run([py, "scripts/ci_check.py", "--include_output"])

    ok = not missing and ci["ok"] and (output_check["ok"] if output_check else True)
    return {
        "ok": ok,
        "mode": "local_release" if include_output else "source_publish",
        "required_source_files": source_files,
        "missing_source_files": missing,
        "ci_check": ci,
        "output_check": output_check,
        "notes": [
            "Source publish mode ignores large generated output folders.",
            "Local release mode also checks output/toy when --include_output is used.",
            "Heavy browser render QA is intentionally opt-in and not part of default release readiness.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_output", action="store_true")
    args = parser.parse_args()
    result = release_readiness(include_output=args.include_output)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
