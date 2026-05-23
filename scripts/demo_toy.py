#!/usr/bin/env python3
"""Reproduce the toy bear demo from existing workspace data.

This script is intentionally explicit so new users can inspect each stage.
It assumes COLMAP and extracted frames already exist under workspace_video/toy.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_object_masks import build_masks
from scripts.cluster_clean_ply import clean_ply
from scripts.crop_aligned_object import crop_aligned
from scripts.publish_output import publish
from sr_3dgs.step3_intrinsic_align import IntrinsicAligner
from sr_3dgs.step4_train_3dgs import SR3DGSTrainer, SRTrainConfig
from sr_3dgs.step5_cleanup import CleanupProcessor
from sr_3dgs.sog_export import export_sog_viewer
from sr_3dgs.splat_export import export_from_ply
from sr_3dgs.web_viewer import generate_viewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="toy")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--bbox", default="40,150,680,960")
    parser.add_argument("--mask_rect", default="40,20,610,720")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scene_root = ROOT / "workspace_video" / args.scene
    aligned_fixed = scene_root / "aligned_fixed"
    aligned_crop = scene_root / "aligned_fixed_toy_bbox"
    mask_root = scene_root / "aligned_fixed_toy_masked" / "masks"
    train_root = scene_root / "train_masked_default_2500"
    package_root = scene_root / "package_masked_v2"

    IntrinsicAligner(
        scene_root / "colmap" / "sparse" / "0",
        scene_root / "sr_images",
        aligned_fixed,
        scale_factor=1,
    ).run(force=True)

    bbox = tuple(int(v) for v in args.bbox.split(","))
    crop_aligned(aligned_fixed, aligned_crop, bbox=bbox)
    build_masks(aligned_crop, mask_root, rect=args.mask_rect, force=True, method="fast")

    SR3DGSTrainer(
        SRTrainConfig(
            data_dir=str(aligned_crop),
            result_dir=str(train_root),
            max_steps=args.steps,
            save_steps=args.steps,
            eval_steps=max(1, args.steps // 2),
            warmup_steps=500,
            sh_degree=2,
            strategy="default",
            max_render_dim=960,
            device=args.device,
            mask_dir=str(mask_root),
            mask_background_weight=0.02,
            mask_alpha_reg=0.08,
        )
    ).run()

    ckpt = train_root / f"checkpoint_step{args.steps}.pt"
    clean_root = package_root / "clean"
    CleanupProcessor(str(ckpt), str(clean_root), device="cpu").run(
        opacity_thresholds=(0.10, 0.50)
    )
    standard_ply = clean_root / "clean_opa0.10_standard.ply"
    preview_ply = clean_root / "clean_opa0.50_standard.ply"
    clean_report = package_root / "cluster_clean.json"
    cluster_ply = clean_root / "clean_opa0.10_standard_cluster.ply"
    clean_ply(
        standard_ply,
        cluster_ply,
        report_path=clean_report,
        opacity_min=0.12,
        voxel_size=0.10,
        max_radius=3.2,
        select="nearest",
        seed=(0.0, 0.0, 0.0),
        min_component_points=800,
        dilate=1,
        auto=True,
    )
    standard_ply = cluster_ply
    splat = package_root / "toy_masked_v2.splat"
    preview_splat = package_root / "toy_masked_v2_preview_opa0.50.splat"
    export_from_ply(str(standard_ply), str(splat), sort_by_depth=True)
    export_from_ply(str(preview_ply), str(preview_splat), sort_by_depth=True)
    generate_viewer(str(splat), str(package_root / "toy_masked_v2_viewer.html"), embed_splat=False)
    export_sog_viewer(
        str(standard_ply),
        str(package_root / "toy_masked_v2_sog.html"),
        overwrite=True,
        unbundled=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_splat_preview.py"),
            str(preview_splat),
            "--out_dir",
            str(package_root / "preview"),
        ],
        check=False,
    )
    publish(package_root, ROOT / "output" / "toy_v2", "toy_v2")
    print("Toy demo published to output/toy_v2")


if __name__ == "__main__":
    main()
