#!/usr/bin/env python3
"""Small smoke tests for scene input quality assessment."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from assess_scene_inputs import assess_scene


def _write_rgb(path, value):
    arr = np.full((48, 64, 3), value, dtype=np.uint8)
    arr[8:28, 12:42] = 255 - value
    Image.fromarray(arr).save(path)


def _write_mask(path):
    arr = np.zeros((48, 64), dtype=np.uint8)
    arr[10:34, 14:46] = 255
    Image.fromarray(arr).save(path)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        scene = Path(tmp) / "scene"
        images = scene / "subject_bbox" / "images"
        masks = scene / "subject_masked" / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for idx in range(6):
            name = f"frame_{idx:03d}.png"
            _write_rgb(images / name, 40 + idx * 20)
            _write_mask(masks / name)

        report = assess_scene(scene, max_frames=6)
        assert report["images"]["path"].endswith("subject_bbox/images"), report
        assert report["images"]["count"] == 6, report
        assert report["masks"]["path"].endswith("subject_masked/masks"), report
        assert report["masks"]["count"] == 6, report
        assert 0 <= report["verdict"]["score"] <= 100, report


if __name__ == "__main__":
    main()
