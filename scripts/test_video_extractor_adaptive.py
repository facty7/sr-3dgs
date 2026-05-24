#!/usr/bin/env python3
"""Unit tests for adaptive video frame selection."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from sr_3dgs.video_extractor import select_frames_adaptive


def _write_frame(path, idx):
    arr = np.zeros((72, 96, 3), dtype=np.uint8)
    arr[:, :] = 40 + idx
    cv2.circle(arr, (24 + idx, 36), 12, (220, 220, 220), -1)
    cv2.rectangle(arr, (8, 8), (14 + idx % 8, 18), (120, 180, 240), -1)
    cv2.imwrite(str(path), arr)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames = []
        for idx in range(12):
            path = root / f"raw_{idx:03d}.png"
            _write_frame(path, idx)
            frames.append(path)

        strict, strict_manifest = select_frames_adaptive(
            frames,
            min_sharpness=2_500.0,
            min_frame_diff=0.0,
            max_frames=12,
            min_frames=8,
            adaptive=False,
        )
        relaxed, relaxed_manifest = select_frames_adaptive(
            frames,
            min_sharpness=2_500.0,
            min_frame_diff=0.0,
            max_frames=12,
            min_frames=8,
            adaptive=True,
        )

        assert len(strict) < 8, strict_manifest
        assert len(relaxed) >= len(strict), relaxed_manifest
        assert relaxed_manifest["relaxed"], relaxed_manifest
        assert relaxed_manifest["selected_pass"].startswith("coverage_"), relaxed_manifest
        assert len(relaxed_manifest["passes"]) > 1, relaxed_manifest
        assert relaxed_manifest["selected_raw_files"], relaxed_manifest


if __name__ == "__main__":
    main()
