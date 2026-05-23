#!/usr/bin/env python3
"""Build a lightweight HTML review page for cleanup candidates."""

import argparse
import html
import json
import os
from pathlib import Path


def _rel(path, base):
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_review(comparison_json, out_html, images, scene="scene"):
    comparison_json = Path(comparison_json)
    out_html = Path(out_html)
    comparison = _load_json(comparison_json)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    image_cards = []
    for label, image_path in images:
        image_path = Path(image_path)
        src = _rel(image_path, out_html.parent)
        image_cards.append(
            f"""      <figure>
        <img src="{html.escape(src)}" alt="{html.escape(label)}">
        <figcaption>{html.escape(label)}</figcaption>
      </figure>"""
        )

    base = comparison["base"]
    rows = [
        (
            "current",
            base["point_count"],
            base["size_mb"],
            base["radius_p99"],
            base.get("scale_actual_p99", base.get("scale_p99")),
            "",
        )
    ]
    for item in comparison["candidates"]:
        delta = item.get("delta_from_base", {})
        rows.append(
            (
                Path(item["path"]).parent.name or Path(item["path"]).stem,
                item["point_count"],
                item["size_mb"],
                item["radius_p99"],
                item.get("scale_actual_p99", item.get("scale_p99")),
                f"{delta.get('points_delta_percent', 0):+.2f}% points",
            )
        )
    table_rows = "\n".join(
        f"""      <tr>
        <td>{html.escape(str(name))}</td>
        <td>{_fmt(points, 0)}</td>
        <td>{_fmt(size, 2)} MB</td>
        <td>{_fmt(radius)}</td>
        <td>{_fmt(scale)}</td>
        <td>{html.escape(note)}</td>
      </tr>"""
        for name, points, size, radius, scale, note in rows
    )

    recommendation = comparison.get("recommended_candidate", {})
    rec_path = recommendation.get("path", "")
    rec_text = (
        f"{rec_path} (score {_fmt(recommendation.get('score'), 2)})"
        if rec_path
        else "No recommendation"
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(scene)} Candidate Review</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #10131a; color: #f5f7fb; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 36px 22px 52px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    h2 {{ margin-top: 34px; font-size: 21px; }}
    p {{ color: #b9c0ce; line-height: 1.6; }}
    code {{ color: #d6e8ff; }}
    .summary {{ border: 1px solid #30394d; border-radius: 8px; padding: 16px; background: #171c27; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; border: 1px solid #30394d; border-radius: 8px; background: #151a24; overflow: hidden; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 10px 12px; color: #d8deec; font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #2d3446; padding: 10px 8px; text-align: left; }}
    th {{ color: #dfe6f5; }}
    td {{ color: #c2cadb; }}
    pre {{ white-space: pre-wrap; border: 1px solid #30394d; border-radius: 8px; padding: 14px; background: #0d1118; color: #d6e8ff; overflow: auto; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(scene)} Candidate Review</h1>
  <section class="summary">
    <p><strong>Recommended candidate:</strong> <code>{html.escape(rec_text)}</code></p>
    <p>{html.escape(recommendation.get("reason", "Geometry/statistics comparison only; visual review is still required."))}</p>
  </section>

  <h2>CPU Contact Sheets</h2>
  <div class="grid">
{chr(10).join(image_cards)}
  </div>

  <h2>Metrics</h2>
  <table>
    <thead>
      <tr><th>Candidate</th><th>Points</th><th>Size</th><th>p99 radius</th><th>p99 actual scale</th><th>Delta</th></tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>

  <h2>Next Heavy Step</h2>
  <pre>PYTHONDONTWRITEBYTECODE=1 python scripts/preflight_heavy.py \\
  {html.escape(rec_path)} \\
  --step sog

PYTHONDONTWRITEBYTECODE=1 python scripts/publish_clean_candidate.py \\
  {html.escape(rec_path)} \\
  --out output/{html.escape(scene)} \\
  --scene_name {html.escape(scene)} \\
  --asset_name {html.escape(scene)} \\
  --replace \\
  --convert_sog \\
  --i_understand_this_is_heavy</pre>

  <p>This page is lightweight review evidence. It does not replace WebGPU/SOG visual QA.</p>
</main>
</body>
</html>
"""
    out_html.write_text(html_text, encoding="utf-8")
    return {"ok": True, "out": str(out_html), "recommended_candidate": rec_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="scene")
    parser.add_argument(
        "--image",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        default=[],
        help="Image label and path. Can be repeated.",
    )
    args = parser.parse_args()
    result = build_review(args.comparison, args.out, args.image, args.scene)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
