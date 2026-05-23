#!/usr/bin/env python3
"""Crop an aligned scene around the object and update camera intrinsics."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.utils import ensure_dir


def _content_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    dark = gray < 115
    texture = cv2.absdiff(gray, blur) > 10
    sat = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1] > 18
    mask = (dark & (texture | sat)).astype(np.uint8) * 255
    h, w = mask.shape
    mask[: int(h * 0.08), :] = 0
    mask[int(h * 0.90) :, :] = 0
    mask[:, : int(w * 0.05)] = 0
    mask[:, int(w * 0.95) :] = 0
    kernel = np.ones((13, 13), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=3)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return None
    min_area = h * w * 0.01
    comps = []
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area >= min_area:
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            center_bias = abs(cx - w * 0.5) / w + abs(cy - h * 0.48) / h
            comps.append((area / (1.0 + center_bias), x, y, bw, bh))
    if not comps:
        return None
    _, x, y, bw, bh = max(comps)
    return x, y, x + bw, y + bh


def _median_bbox(images_dir, names, pad_ratio):
    boxes = []
    for name in names:
        path = images_dir / name
        if not path.exists():
            continue
        img = np.array(Image.open(path).convert("RGB"))
        bbox = _content_bbox(img)
        if bbox is not None:
            boxes.append(bbox)
    if not boxes:
        raise RuntimeError("Could not infer an object crop from the aligned images.")
    arr = np.array(boxes, dtype=np.float32)
    x0, y0, x1, y1 = np.percentile(arr, [15, 15, 85, 85], axis=0).diagonal()
    h, w = np.array(Image.open(images_dir / names[0]).convert("RGB")).shape[:2]
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    size = max(x1 - x0, y1 - y0)
    size *= 1.0 + pad_ratio * 2.0
    crop_w = min(w, int(round(size)))
    crop_h = min(h, int(round(size * 1.15)))
    left = max(0, min(w - crop_w, int(round(cx - crop_w * 0.5))))
    top = max(0, min(h - crop_h, int(round(cy - crop_h * 0.52))))
    return left, top, left + crop_w, top + crop_h


def crop_aligned(aligned, out, pad_ratio=0.18, bbox=None):
    aligned = Path(aligned)
    out = Path(out)
    ensure_dir(out)
    images_out = out / "images"
    ensure_dir(images_out)

    loaded = np.load(aligned / "scene_data.npz", allow_pickle=True)
    transforms = loaded["transforms"].tolist()
    names = [t["image_name"] for t in transforms]
    images_dir = aligned / "images"
    if bbox is None:
        left, top, right, bottom = _median_bbox(images_dir, names, pad_ratio)
    else:
        left, top, right, bottom = bbox
    crop_w = right - left
    crop_h = bottom - top

    cropped_transforms = []
    for t in transforms:
        src = images_dir / t["image_name"]
        dst = images_out / t["image_name"]
        if src.exists():
            img = Image.open(src).convert("RGB")
            img.crop((left, top, right, bottom)).save(dst)
        ct = dict(t)
        K = np.array(ct["K"], dtype=np.float32)
        K[0, 2] -= left
        K[1, 2] -= top
        ct["K"] = K.tolist()
        ct["width"] = crop_w
        ct["height"] = crop_h
        cropped_transforms.append(ct)

    np.savez(
        out / "scene_data.npz",
        transforms=cropped_transforms,
        points_xyz=loaded["points_xyz"],
        points_rgb=loaded["points_rgb"],
        scale_factor_w=loaded.get("scale_factor_w", 1.0),
        scale_factor_h=loaded.get("scale_factor_h", 1.0),
    )
    meta = {
        "source": str(aligned),
        "crop": {"left": left, "top": top, "right": right, "bottom": bottom},
        "width": crop_w,
        "height": crop_h,
        "num_images": len(cropped_transforms),
        "num_points": int(len(loaded["points_xyz"])),
    }
    (out / "crop_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if (aligned / "metadata.json").exists():
        shutil.copy2(aligned / "metadata.json", out / "metadata_source.json")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pad_ratio", type=float, default=0.18)
    parser.add_argument(
        "--bbox",
        help="Optional explicit crop as left,top,right,bottom in aligned image pixels.",
    )
    args = parser.parse_args()
    bbox = tuple(int(v) for v in args.bbox.split(",")) if args.bbox else None
    if bbox is not None and len(bbox) != 4:
        raise ValueError("--bbox must be left,top,right,bottom")
    meta = crop_aligned(args.aligned, args.out, args.pad_ratio, bbox=bbox)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
