#!/usr/bin/env python3
"""Lightweight frame/mask quality report for object-centric scenes."""

import argparse
import html as html_lib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _image_files(path):
    path = Path(path)
    if not path.exists():
        return []
    if (path / "images").exists():
        path = path / "images"
    return [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _mask_files(path):
    path = Path(path)
    if not path or not path.exists():
        return []
    if (path / "masks").exists():
        path = path / "masks"
    return [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def _find_default_images(scene_dir):
    scene = Path(scene_dir)
    preferred = [
        scene / "aligned_object" / "images",
        scene / "aligned_object",
        scene / "aligned_fixed_bbox" / "images",
        scene / "aligned_fixed_bbox",
        scene / "aligned_fixed" / "images",
        scene / "aligned_fixed",
        scene / "aligned_bbox" / "images",
        scene / "aligned_bbox",
        scene / "aligned" / "images",
        scene / "aligned",
        scene / "frames",
    ]
    for path in preferred:
        if path.exists() and _image_files(path):
            return path
    matches = []
    for child in scene.glob("*/images"):
        files = _image_files(child)
        if files:
            name = child.parent.name.lower()
            priority = 2 if ("bbox" in name or "object" in name) else 1 if "aligned" in name else 0
            matches.append((priority, len(files), str(child), child))
    if matches:
        return sorted(matches, reverse=True)[0][3]
    raise FileNotFoundError(f"No image set found under {scene}")


def _find_default_masks(scene_dir):
    scene = Path(scene_dir)
    preferred = [
        scene / "aligned_object_masked" / "masks",
        scene / "aligned_object" / "masks",
        scene / "aligned_fixed_bbox" / "masks",
        scene / "aligned_fixed" / "masks",
        scene / "aligned_bbox" / "masks",
        scene / "aligned" / "masks",
    ]
    for path in preferred:
        if path.exists() and _mask_files(path):
            return path
    matches = []
    for child in scene.glob("*/masks"):
        files = _mask_files(child)
        if files:
            name = child.parent.name.lower()
            priority = 3 if "masked" in name else 2 if ("bbox" in name or "object" in name) else 1 if "aligned" in name else 0
            matches.append((priority, len(files), str(child), child))
    return sorted(matches, reverse=True)[0][3] if matches else None


def _read_gray(path, max_side=512):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray


def _image_size(path):
    with Image.open(path) as img:
        return [int(img.size[0]), int(img.size[1])]


def _sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _diff(prev, cur):
    if prev is None:
        return None
    if prev.shape != cur.shape:
        cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]))
    return float(np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32))) / 255.0)


def _mask_stats(mask_path):
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    fg = arr > 127
    ratio = float(np.mean(fg))
    if not fg.any():
        return {"fg_ratio": ratio, "bbox_fill": 0.0, "touches_edge": False}
    ys, xs = np.where(fg)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    touches = x0 <= 2 or y0 <= 2 or x1 >= arr.shape[1] - 3 or y1 >= arr.shape[0] - 3
    return {
        "fg_ratio": ratio,
        "bbox_fill": float(fg.sum() / bbox_area),
        "touches_edge": bool(touches),
    }


def _percentiles(values):
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {}
    arr = np.array(clean, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _score_and_recommend(frame_count, sharp, diffs, masks):
    problems = []
    recommendations = []
    score = 100

    if frame_count < 48:
        score -= 20
        problems.append("low_frame_count")
        recommendations.append("Extract more usable frames or capture a slower 360 pass.")
    elif frame_count < 90:
        score -= 8
        recommendations.append("More frames may help thin or symmetric objects.")

    if sharp.get("p10", 999.0) < 35:
        score -= 18
        problems.append("blurry_frames")
        recommendations.append("Raise min sharpness or reshoot with slower motion and locked focus.")
    elif sharp.get("p10", 999.0) < 80:
        score -= 8
        recommendations.append("Consider filtering the blurriest frames before training.")

    if diffs.get("p50", 1.0) < 0.015:
        score -= 10
        problems.append("near_duplicate_frames")
        recommendations.append("Increase frame-difference filtering to avoid many repeated views.")
    if diffs.get("p90", 0.0) > 0.22:
        score -= 12
        problems.append("large_view_jumps_or_exposure_changes")
        recommendations.append("Use steadier capture or lower FPS sampling to reduce sudden changes.")

    if masks["count"]:
        fg = masks["fg_ratio"]
        if fg.get("p50", 0.0) < 0.08:
            score -= 14
            problems.append("mask_too_small")
            recommendations.append("Use a larger foreground rectangle or better mask backend.")
        if fg.get("p50", 1.0) > 0.75:
            score -= 14
            problems.append("mask_too_large")
            recommendations.append("Tighten crop/mask so background is not treated as the object.")
        if masks["edge_touch_percent"] > 35:
            score -= 10
            problems.append("mask_touches_edges")
            recommendations.append("Widen crop or improve masks; edge-cut objects reconstruct poorly.")
    else:
        score -= 12
        problems.append("missing_masks")
        recommendations.append("Run object masks for object-centric videos before masked training.")

    if not recommendations:
        recommendations.append("Input set looks usable; prefer small controlled parameter sweeps over long blind training.")

    return {
        "score": max(0, min(100, int(round(score)))),
        "problems": problems,
        "recommendations": recommendations,
    }


def assess_scene(scene_dir, images_dir=None, masks_dir=None, max_frames=160):
    scene = Path(scene_dir)
    images_root = Path(images_dir) if images_dir else _find_default_images(scene)
    masks_root = Path(masks_dir) if masks_dir else _find_default_masks(scene)
    images = _image_files(images_root)
    if not images:
        raise FileNotFoundError(f"No images found in {images_root}")

    step = max(1, math.ceil(len(images) / max_frames))
    sampled = images[::step]
    sharpness = []
    diffs = []
    prev = None
    dimensions = []
    for path in sampled:
        dimensions.append(_image_size(path))
        gray = _read_gray(path)
        sharpness.append(_sharpness(gray))
        diffs.append(_diff(prev, gray))
        prev = gray

    mask_summary = {"count": 0}
    if masks_root:
        masks = _mask_files(masks_root)
        image_stems = {img.stem for img in images}
        paired = [m for m in masks if m.stem in image_stems]
        ratios = []
        fills = []
        edge_count = 0
        for path in paired[:: max(1, math.ceil(max(1, len(paired)) / max_frames))]:
            stats = _mask_stats(path)
            ratios.append(stats["fg_ratio"])
            fills.append(stats["bbox_fill"])
            edge_count += int(stats["touches_edge"])
        mask_summary = {
            "path": str(masks_root),
            "count": len(paired),
            "fg_ratio": _percentiles(ratios),
            "bbox_fill": _percentiles(fills),
            "edge_touch_percent": round(100.0 * edge_count / max(1, len(ratios)), 2),
        }

    sharp = _percentiles(sharpness)
    diff_stats = _percentiles(diffs)
    verdict = _score_and_recommend(len(images), sharp, diff_stats, mask_summary)
    return {
        "scene": str(scene),
        "images": {
            "path": str(images_root),
            "count": len(images),
            "sampled_count": len(sampled),
            "dimensions_first_sample": dimensions[0] if dimensions else None,
            "sharpness_laplacian": sharp,
            "frame_diff": diff_stats,
        },
        "masks": mask_summary,
        "verdict": verdict,
    }


def write_html(report, path):
    path = Path(path)
    recs = "\n".join(
        f"<li>{html_lib.escape(item)}</li>" for item in report["verdict"]["recommendations"]
    )
    problems = report["verdict"]["problems"] or ["none"]
    problem_text = html_lib.escape(", ".join(problems))
    scene_text = html_lib.escape(report["scene"])
    report_text = html_lib.escape(json.dumps(report, indent=2))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scene Input Quality</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #101218; color: #f4f6fb; }}
    main {{ max-width: 960px; margin: 0 auto; padding: 36px 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    pre {{ overflow: auto; padding: 16px; background: #181c25; border: 1px solid #2d3341; border-radius: 8px; }}
    .score {{ font-size: 46px; font-weight: 750; }}
    .muted {{ color: #aeb7c7; }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
<main>
  <h1>Scene Input Quality</h1>
  <div class="muted">{scene_text}</div>
  <div class="score">{report["verdict"]["score"]}/100</div>
  <h2>Problems</h2>
  <p>{problem_text}</p>
  <h2>Recommendations</h2>
  <ul>{recs}</ul>
  <h2>Full Report</h2>
  <pre>{report_text}</pre>
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir")
    parser.add_argument("--images", default="")
    parser.add_argument("--masks", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--html", default="")
    args = parser.parse_args()
    report = assess_scene(args.scene_dir, args.images or None, args.masks or None)
    text = json.dumps(report, indent=2)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    if args.html:
        write_html(report, args.html)
    print(text)


if __name__ == "__main__":
    main()
