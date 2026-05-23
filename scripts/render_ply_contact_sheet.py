#!/usr/bin/env python3
"""Render lightweight CPU contact sheets for standard 3DGS PLY files."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.quality import diagnose_file


def _read_ply(path):
    with open(path, "rb") as f:
        props = []
        count = 0
        while True:
            line = f.readline().decode("utf-8").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float") or line.startswith("property uchar"):
                _, typ, name = line.split()
                props.append((name, {"float": np.float32, "uchar": np.uint8}[typ]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=count)


def _sample(data, max_points):
    if len(data) <= max_points:
        return data
    idx = np.linspace(0, len(data) - 1, max_points).astype(np.int64)
    return data[idx]


def _colors(data):
    names = data.dtype.names or ()
    if all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        rgb = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]])
        rgb = np.clip((rgb + 0.5) * 255.0, 0, 255)
        return rgb.astype(np.uint8)
    if all(name in names for name in ("red", "green", "blue")):
        return np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.uint8)
    return np.full((len(data), 3), 210, dtype=np.uint8)


def _project(xyz, axes, center, radius, size, margin):
    xy = xyz[:, axes] - center[list(axes)]
    scale = (size - margin * 2) / max(radius * 2.0, 1e-6)
    px = (xy[:, 0] * scale + size / 2).astype(np.int32)
    py = (size / 2 - xy[:, 1] * scale).astype(np.int32)
    keep = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    return px[keep], py[keep], keep


def _draw_view(data, title, axes, size, margin, max_points):
    data = _sample(data, max_points)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    color = _colors(data)
    center = np.nanmedian(xyz, axis=0)
    dist = np.linalg.norm(xyz - center, axis=1)
    radius = float(np.percentile(dist, 99))
    px, py, keep = _project(xyz, axes, center, radius, size, margin)

    img = Image.new("RGB", (size, size), (17, 20, 28))
    pixels = img.load()
    kept_colors = color[keep]
    for x, y, c in zip(px, py, kept_colors):
        pixels[int(x), int(y)] = tuple(int(v) for v in c)

    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(54, 62, 80))
    draw.text((12, 10), title, fill=(238, 241, 247))
    draw.text((12, size - 24), f"{len(data):,} sampled", fill=(178, 186, 204))
    return img


def render_ply(path, out, title="", size=520, max_points=180_000):
    path = Path(path)
    data = _read_ply(path)
    panels = [
        _draw_view(data, "front x/z", (0, 2), size, 28, max_points),
        _draw_view(data, "side y/z", (1, 2), size, 28, max_points),
        _draw_view(data, "top x/y", (0, 1), size, 28, max_points),
    ]
    header_h = 74
    gap = 12
    sheet = Image.new("RGB", (size * 3 + gap * 2, size + header_h), (13, 15, 21))
    draw = ImageDraw.Draw(sheet)
    diag = diagnose_file(path)
    radius = diag.get("radius_percentiles", {})
    scale = diag.get("scale_percentiles", {})
    title = title or path.name
    draw.text((16, 14), title, fill=(244, 246, 250))
    draw.text(
        (16, 40),
        f"{diag['count']:,} points | p99 radius {radius.get('p99', 0):.3f} | p99 scale {scale.get('p99', 0):.3f}",
        fill=(180, 188, 205),
    )
    for i, panel in enumerate(panels):
        sheet.paste(panel, (i * (size + gap), header_h))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return {"path": str(path), "out": str(out), "diagnostics": diag}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--size", type=int, default=520)
    parser.add_argument("--max_points", type=int, default=180_000)
    args = parser.parse_args()
    result = render_ply(args.ply, args.out, args.title, args.size, args.max_points)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
