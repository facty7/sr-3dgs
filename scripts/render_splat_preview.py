#!/usr/bin/env python3
"""Render a robust PNG/GIF preview directly from a .splat file.

This is a diagnostic preview, not a photoreal 3DGS renderer. It projects the
cleaned splat centers as colored surfels so a scene can be inspected even when
WebGL/GPU viewers are unavailable or broken.
"""

import argparse
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_splat(path):
    raw = Path(path).read_bytes()
    count = len(raw) // 32
    pos = np.empty((count, 3), dtype=np.float32)
    scale = np.empty((count, 3), dtype=np.float32)
    rgba = np.empty((count, 4), dtype=np.uint8)
    for i in range(count):
        off = i * 32
        pos[i] = struct.unpack_from("<fff", raw, off)
        scale[i] = struct.unpack_from("<fff", raw, off + 12)
        rgba[i] = struct.unpack_from("<BBBB", raw, off + 24)
    return pos, scale, rgba


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    z = eye - target
    z = z / max(np.linalg.norm(z), 1e-8)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def render_frame(pos, scale, rgba, angle, width, height, bg):
    center = np.median(pos, axis=0)
    dist = np.linalg.norm(pos - center, axis=1)
    radius = max(float(np.percentile(dist, 99.5)), 1e-3)
    cam_dist = radius * 3.2
    eye = center + np.array([
        math.cos(angle) * cam_dist,
        math.sin(angle) * cam_dist,
        cam_dist * 0.35,
    ], dtype=np.float32)
    x_axis, y_axis, z_axis = look_at(eye, center)

    rel = pos - eye
    cam_x = rel @ x_axis
    cam_y = rel @ y_axis
    cam_z = -(rel @ z_axis)

    valid = cam_z > 1e-4
    cam_x, cam_y, cam_z = cam_x[valid], cam_y[valid], cam_z[valid]
    colors = rgba[valid, :3].astype(np.float32)
    alpha = rgba[valid, 3].astype(np.float32) / 255.0
    splat_scale = scale[valid].mean(axis=1)

    focal = min(width, height) * 0.95
    px = width * 0.5 + focal * cam_x / cam_z
    py = height * 0.5 - focal * cam_y / cam_z
    size = np.clip((splat_scale * focal / cam_z) * 3.0, 1.0, 6.0)

    in_view = (
        (px >= -20) & (px < width + 20) &
        (py >= -20) & (py < height + 20)
    )
    px, py, cam_z = px[in_view], py[in_view], cam_z[in_view]
    colors, alpha, size = colors[in_view], alpha[in_view], size[in_view]

    order = np.argsort(-cam_z)
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:] = np.array(bg, dtype=np.float32)

    for idx in order:
        x0 = int(round(px[idx]))
        y0 = int(round(py[idx]))
        r = int(max(1, round(size[idx])))
        x1, x2 = max(0, x0 - r), min(width, x0 + r + 1)
        y1, y2 = max(0, y0 - r), min(height, y0 + r + 1)
        if x1 >= x2 or y1 >= y2:
            continue
        patch = img[y1:y2, x1:x2]
        a = min(0.95, max(0.15, float(alpha[idx])))
        patch[:] = patch * (1.0 - a) + colors[idx] * a

    out = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def add_label(img, text):
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, 8 + 7 * len(text), 30), fill=(10, 12, 20))
    draw.text((12, 12), text, fill=(230, 232, 240))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("splat")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gif", action="store_true", default=True)
    args = parser.parse_args()

    splat = Path(args.splat)
    out_dir = Path(args.out_dir) if args.out_dir else splat.parent / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    pos, scale, rgba = read_splat(splat)
    frames = []
    for i in range(args.frames):
        angle = 2.0 * math.pi * i / args.frames
        img = render_frame(pos, scale, rgba, angle, args.width, args.height, bg=(18, 19, 30))
        img = add_label(img, f"{splat.name}  frame {i + 1}/{args.frames}")
        frame_path = out_dir / f"frame_{i:03d}.png"
        img.save(frame_path)
        frames.append(img)

    cols = 4
    rows = math.ceil(min(len(frames), 12) / cols)
    thumb_w = args.width // 2
    thumb_h = args.height // 2
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (18, 19, 30))
    for i, img in enumerate(frames[:12]):
        thumb = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, ((i % cols) * thumb_w, (i // cols) * thumb_h))
    sheet_path = out_dir / "contact_sheet.png"
    sheet.save(sheet_path)

    gif_path = out_dir / "turntable.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )
    print(sheet_path)
    print(gif_path)


if __name__ == "__main__":
    main()
