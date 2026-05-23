#!/usr/bin/env python3
"""Lightweight repository audit for open-source readiness."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BLOCKED_SUFFIXES = {".pyc", ".pyo"}
BLOCKED_NAMES = {"__pycache__", ".DS_Store"}
IGNORED_ROOTS = {"output", "workspace_video", "workspace", "delivery", "input_videos"}


def _find_blocked(root):
    items = []
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if ".git/" in rel or rel.startswith(".git/"):
            continue
        if rel.split("/", 1)[0] in IGNORED_ROOTS:
            continue
        if path.name in BLOCKED_NAMES or path.suffix in BLOCKED_SUFFIXES or path.name.endswith(".bak"):
            items.append(rel)
    return sorted(items)


def _missing_files(root):
    required = ["README.md", "requirements.txt", "requirements-optional.txt", "setup.py"]
    return [name for name in required if not (root / name).exists()]


def audit(root=ROOT):
    root = Path(root)
    blocked = _find_blocked(root)
    missing = _missing_files(root)
    problems = []
    if blocked:
        problems.append("remove generated/cache/backup files")
    if missing:
        problems.append("missing basic project files")
    return {
        "root": str(root),
        "ok": not problems,
        "problems": problems,
        "blocked_files": blocked,
        "missing_files": missing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
