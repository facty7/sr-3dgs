#!/usr/bin/env python3
"""Parse Python files without writing __pycache__ files."""

import argparse
import ast
import json
from pathlib import Path


def check(paths):
    failures = []
    checked = 0
    for item in paths:
        path = Path(item)
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            if "__pycache__" in file.parts:
                continue
            checked += 1
            try:
                ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            except SyntaxError as exc:
                failures.append({
                    "file": str(file),
                    "line": exc.lineno,
                    "message": exc.msg,
                })
    return {"ok": not failures, "checked": checked, "failures": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["scripts", "sr_3dgs"])
    args = parser.parse_args()
    result = check(args.paths)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
