#!/usr/bin/env python3
"""Build foreground masks for aligned images.

This is a pragmatic first pass for object-centric scenes: it uses GrabCut with
an optional rectangle prior, then keeps the main foreground component.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.utils import ensure_dir


def _parse_rect(rect, w, h):
    if rect:
        x0, y0, x1, y1 = [int(v) for v in rect.split(",")]
    else:
        x0 = int(w * 0.08)
        y0 = int(h * 0.03)
        x1 = int(w * 0.92)
        y1 = int(h * 0.88)
    x0 = max(0, min(w - 2, x0))
    y0 = max(0, min(h - 2, y0))
    x1 = max(x0 + 1, min(w - 1, x1))
    y1 = max(y0 + 1, min(h - 1, y1))
    return x0, y0, x1 - x0, y1 - y0


def _largest_component(mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return labels == keep


def build_mask(image, rect=None, iterations=5, method="fast"):
    if method == "rembg":
        try:
            from rembg import remove
            rgba = remove(image.convert("RGB"))
            arr = np.array(rgba.convert("RGBA"))
            alpha = arr[:, :, 3]
            fg = alpha > 8
            fg = _largest_component(fg)
            kernel = np.ones((9, 9), np.uint8)
            fg = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
            return cv2.GaussianBlur(fg, (9, 9), 0)
        except Exception as exc:
            print(f"[Mask] rembg unavailable/failed ({exc}); falling back to fast")
            method = "fast"

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if method == "fast":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        blur = cv2.GaussianBlur(gray, (0, 0), 5)
        texture = cv2.absdiff(gray, blur) > 7
        dark = gray < 150
        sat = hsv[:, :, 1] > 18
        fg = (dark & (texture | sat))
        if rect:
            x, y, rw, rh = _parse_rect(rect, w, h)
            prior = np.zeros((h, w), dtype=bool)
            prior[y:y + rh, x:x + rw] = True
            fg &= prior
        fg[: int(h * 0.02), :] = False
        fg[int(h * 0.96) :, :] = False
        fg = _largest_component(fg)
        kernel = np.ones((17, 17), np.uint8)
        fg = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.dilate(fg, kernel, iterations=2)
        return cv2.GaussianBlur(fg, (13, 13), 0)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, _parse_rect(rect, w, h), bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    fg = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)
    fg = _largest_component(fg)
    kernel = np.ones((9, 9), np.uint8)
    fg = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    fg = cv2.GaussianBlur(fg, (9, 9), 0)
    return fg


def build_masks(aligned_dir, out_dir=None, rect=None, force=False, method="fast"):
    aligned = Path(aligned_dir)
    out = Path(out_dir) if out_dir else aligned / "masks"
    ensure_dir(out)
    images = sorted((aligned / "images").glob("*"))
    written = []
    for path in images:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        dst = out / (path.stem + ".png")
        if dst.exists() and not force:
            written.append(dst)
            continue
        mask = build_mask(Image.open(path), rect=rect, method=method)
        Image.fromarray(mask).save(dst)
        written.append(dst)
    manifest = {
        "source": str(aligned),
        "mask_dir": str(out),
        "count": len(written),
        "rect": rect or "auto",
        "method": method,
    }
    (out / "mask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--rect", default="", help="Optional x0,y0,x1,y1 foreground prior in image pixels")
    parser.add_argument("--method", choices=("fast", "grabcut", "rembg"), default="fast")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_masks(args.aligned, args.out or None, args.rect or None, args.force, args.method), indent=2))


if __name__ == "__main__":
    main()
