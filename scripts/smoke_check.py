#!/usr/bin/env python3
"""Run the lightweight checks expected before sharing the project."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(name, cmd):
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    return {
        "name": name,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/toy")
    parser.add_argument("--benchmark_config", default="configs/benchmark_outputs.json")
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--preview_url", default="")
    parser.add_argument("--skip_preview_url", action="store_true")
    parser.add_argument("--require_preview_url", action="store_true",
                        help="Fail if preview_url is unreachable. Default records it as optional.")
    parser.add_argument("--render_preview_url", default="")
    parser.add_argument("--render_screenshot", default="workspace_video/qa/smoke_render.png")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument(
        "--enable_heavy_render_qa",
        action="store_true",
        help="Opt in to launching Chrome/Edge for WebGL screenshot QA. This can be heavy.",
    )
    args = parser.parse_args()

    py = sys.executable
    checks = [
        ("syntax", [py, "scripts/check_syntax.py", "scripts", "sr_3dgs"]),
        ("cluster_clean_unit", [py, "scripts/test_cluster_clean.py"]),
        ("repo_audit", [py, "scripts/audit_repo.py"]),
        ("validate_output", [py, "scripts/validate_output.py", args.output]),
        ("score_output", [py, "scripts/score_output.py", args.output]),
    ]
    if args.benchmark_config and not args.skip_benchmark:
        checks.append(("benchmark_outputs", [py, "scripts/benchmark_outputs.py", "--config", args.benchmark_config]))
    if args.preview_url and not args.skip_preview_url:
        name = "preview_qa" if args.require_preview_url else "preview_qa_optional"
        checks.append((name, [py, "scripts/qa_preview.py", args.preview_url]))
    if args.render_preview_url and args.enable_heavy_render_qa and not args.skip_render:
        checks.append((
            "render_qa",
            [
                py,
                "scripts/qa_render_chrome.py",
                args.render_preview_url,
                "--out",
                args.render_screenshot,
                "--allow_skip",
                "--enable_heavy_browser",
            ],
        ))

    results = [_run(name, cmd) for name, cmd in checks]
    required = [item for item in results if not item["name"].endswith("_optional")]
    report = {"ok": all(item["ok"] for item in required), "checks": results}
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
