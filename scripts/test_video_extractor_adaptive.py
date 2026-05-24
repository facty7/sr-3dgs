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


def _write_sharp_frame(path, idx):
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    offset = idx % 8
    arr[:, :] = 20 + offset
    cv2.rectangle(arr, (8 + offset, 8), (55 + offset, 55), (230, 230, 230), -1)
    cv2.line(arr, (0, idx % 80), (79, (idx * 3) % 80), (40, 180, 240), 2)
    cv2.putText(
        arr,
        str(idx % 100),
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), arr)


def _write_span_frame(path, idx, sharp=True):
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:, :] = 30 + (idx % 25)
    cv2.rectangle(arr, (8, 8), (62, 62), (220, 220, 220), -1)
    cv2.line(arr, (0, (idx * 7) % 80), (79, (idx * 11) % 80), (20, 130, 240), 2)
    if sharp:
        cv2.putText(
            arr,
            str(idx),
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            arr,
            str(idx),
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        arr = cv2.GaussianBlur(arr, (5, 5), 0)
    cv2.imwrite(str(path), arr)


def _write_health_frame(path, idx):
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    if idx == 0:
        arr[:, :] = 0
    elif idx == 1:
        arr[:, :] = 255
    elif idx == 2:
        arr[:, :] = 128
    else:
        arr[:, :] = 30 + idx
        cv2.rectangle(arr, (8, 8), (62, 62), (220, 220, 220), -1)
        cv2.line(arr, (0, (idx * 5) % 80), (79, (idx * 9) % 80), (20, 130, 240), 2)
        cv2.putText(
            arr,
            str(idx),
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
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

        long_frames = []
        for idx in range(30):
            path = root / f"long_{idx:03d}.png"
            _write_sharp_frame(path, idx)
            long_frames.append(path)

        selected, manifest = select_frames_adaptive(
            long_frames,
            min_sharpness=1.0,
            min_frame_diff=0.0,
            max_frames=6,
            min_frames=6,
            adaptive=False,
        )

        selected_names = [path.name for path in selected]
        selected_indices = [int(name.split("_")[1].split(".")[0]) for name in selected_names]
        assert len(selected) == 6, manifest
        assert selected_indices[0] == 0, selected_indices
        assert selected_indices[-1] == 29, selected_indices
        assert any(idx > 20 for idx in selected_indices), selected_indices
        strict_pass = manifest["passes"][0]
        assert strict_pass["eligible_count"] == 30, strict_pass
        assert strict_pass["temporal_thinned_count"] == 24, strict_pass
        assert strict_pass["selected_raw_index_coverage"] == 1.0, strict_pass

        span_frames = []
        for idx in range(30):
            path = root / f"span_{idx:03d}.png"
            _write_span_frame(path, idx, sharp=idx < 16)
            span_frames.append(path)

        partial, partial_manifest = select_frames_adaptive(
            span_frames,
            min_sharpness=800.0,
            min_frame_diff=0.0,
            max_frames=30,
            min_frames=8,
            min_span=0.80,
            adaptive=False,
        )
        assert len(partial) >= 8, partial_manifest
        assert partial_manifest["selected_meets_frame_target"] is True, partial_manifest
        assert partial_manifest["selected_meets_span_target"] is False, partial_manifest
        assert partial_manifest["selected_span"] < 0.80, partial_manifest

        covered, covered_manifest = select_frames_adaptive(
            span_frames,
            min_sharpness=800.0,
            min_frame_diff=0.0,
            max_frames=30,
            min_frames=8,
            min_span=0.80,
            adaptive=True,
        )
        assert len(covered) >= 8, covered_manifest
        assert covered_manifest["relaxed"], covered_manifest
        assert covered_manifest["selected_pass"].startswith("coverage_"), covered_manifest
        assert covered_manifest["selected_meets_target"] is True, covered_manifest
        assert covered_manifest["selected_span"] >= 0.80, covered_manifest

        health_frames = []
        for idx in range(12):
            path = root / f"health_{idx:03d}.png"
            _write_health_frame(path, idx)
            health_frames.append(path)

        healthy, health_manifest = select_frames_adaptive(
            health_frames,
            min_sharpness=1.0,
            min_frame_diff=0.0,
            max_frames=12,
            min_frames=6,
            min_span=0.0,
            adaptive=False,
        )
        assert len(healthy) == 9, health_manifest
        assert health_manifest["selected_skipped_bad_exposure"] == 3, health_manifest
        selected_names = {path.name for path in healthy}
        assert "health_000.png" not in selected_names, selected_names
        assert "health_001.png" not in selected_names, selected_names
        assert "health_002.png" not in selected_names, selected_names


if __name__ == "__main__":
    main()
