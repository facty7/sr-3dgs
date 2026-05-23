#!/usr/bin/env python3
"""Benchmark already-published output folders.

This is intentionally lightweight: it does not train, convert, download, or
launch a browser. It scores final deliveries so multiple scenes can be tracked
with the same gates.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_output import _score_delivery


def run_benchmark(config_path):
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    gates = cfg.get("gates", {})
    mobile_sog_mb = float(gates.get("mobile_sog_mb", 12.0))
    min_points = int(gates.get("min_points", 120000))
    min_score = int(gates.get("min_score", 85))
    results = []
    for scene in cfg.get("scenes", []):
        score = _score_delivery(
            scene["output"],
            mobile_sog_mb=mobile_sog_mb,
            min_points=min_points,
        )
        score["name"] = scene.get("name", Path(scene["output"]).name)
        score["notes"] = scene.get("notes", "")
        score["meets_min_score"] = score["score"] >= min_score
        results.append(score)

    ok = all(item["ok"] and item["meets_min_score"] for item in results)
    return {
        "config": str(config_path),
        "ok": ok,
        "gates": {
            "mobile_sog_mb": mobile_sog_mb,
            "min_points": min_points,
            "min_score": min_score,
        },
        "scene_count": len(results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_outputs.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = run_benchmark(args.config)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
