#!/usr/bin/env python3
"""CLI entry point for the SR + 3DGS pipeline.

Usage:
    # Full pipeline with defaults
    python run_pipeline.py --input_dir /path/to/images

    # With specific SR model
    python run_pipeline.py --input_dir /path/to/images --sr_mode model --sr_model real-esrgan --sr_scale 4

    # Disable learned super-resolution
    python run_pipeline.py --input_dir /path/to/images --sr_mode off --sr_scale 1

    # Only run steps 1-3 (COLMAP + SR + alignment)
    python run_pipeline.py --input_dir /path/to/images --start 1 --end 3

    # Resume from step 4 using existing aligned data
    python run_pipeline.py --work_dir workspace --start 4 --end 5

    # Use SUPIR for extremely degraded input
    python run_pipeline.py --input_dir /path/to/images --sr_model supir --sr_scale 4

    # Batch process from a directory list
    python run_pipeline.py --batch_list scenes.txt
"""

import os
import sys
import argparse
import json

# Ensure sr_3dgs is importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from sr_3dgs.pipeline import Pipeline, PipelineConfig
from sr_3dgs.utils import ensure_dir


def build_parser():
    p = argparse.ArgumentParser(
        description="SR + 3DGS Pipeline — decoupled super-resolution and "
                    "3D Gaussian Splatting for production-quality 3D from "
                    "low-resolution images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Main args
    p.add_argument("--input_dir", type=str, default="",
                   help="Directory containing raw input images")
    p.add_argument("--work_dir", type=str, default="workspace",
                   help="Working directory for all outputs")
    p.add_argument("--config", type=str, default="",
                   help="JSON config file (overrides CLI args)")

    # Step control
    p.add_argument("--start", type=int, default=1,
                   help="Start from step (1-5)")
    p.add_argument("--end", "--end_step", dest="end", type=int, default=6,
                   help="End at step (1-6)")
    p.add_argument("--skip_existing", type=int, default=1,
                   help="Skip steps with existing output (1=yes, 0=no)")

    # SR config
    p.add_argument("--sr_mode", type=str, default="auto",
                   choices=["auto", "off", "resize", "model"],
                   help="SR strategy: auto selects model for scale>1 and copy for scale=1")
    p.add_argument("--sr_model", type=str, default="real-esrgan",
                   choices=["real-esrgan", "dat", "supir", "basicvsr++", "resize", "off"],
                   help="Super-resolution model")
    p.add_argument("--sr_scale", type=int, default=4,
                   choices=[1, 2, 4, 8],
                   help="Super-resolution scale factor")
    p.add_argument("--sr_device", type=str, default="cuda",
                   help="Device for SR inference")
    p.add_argument("--sr_model_load_timeout", type=int, default=180,
                   help="Seconds to wait for learned SR before fallback; <=0 disables")
    p.add_argument("--sr_frame_timeout", type=int, default=300,
                   help="Seconds without SR image progress before fallback; <=0 disables")
    p.add_argument("--sr_strict_model", action="store_true",
                   help="Fail instead of falling back when --sr_mode model cannot run")

    # COLMAP config
    p.add_argument("--colmap_path", type=str, default="colmap",
                   help="Path to COLMAP executable")
    p.add_argument("--colmap_camera", type=str, default="SIMPLE_PINHOLE",
                   help="COLMAP camera model")
    p.add_argument("--colmap_camera_fallbacks", type=str, default="SIMPLE_RADIAL,PINHOLE",
                   help="Comma-separated fallback camera models for weak COLMAP reconstructions")
    p.add_argument("--colmap_min_registered_ratio", type=float, default=0.45,
                   help="Minimum registered-image ratio before trying COLMAP fallbacks")
    p.add_argument("--colmap_min_registered_images", type=int, default=24,
                   help="Minimum registered-image count before trying COLMAP fallbacks")
    p.add_argument("--colmap_gpu", type=int, default=0,
                   help="GPU index for COLMAP")

    # Training config
    p.add_argument("--max_steps", type=int, default=30_000,
                   help="Total training steps")
    p.add_argument("--warmup_steps", type=int, default=1_000,
                   help="Warmup steps with reduced LR")
    p.add_argument("--train_device", type=str, default="cuda",
                   help="Device for 3DGS training")

    # Viewer/Export options
    p.add_argument("--no_viewer", type=int, default=0,
                   help="Skip HTML viewer generation (1=yes, 0=no)")
    p.add_argument("--viewer_title", type=str, default="",
                   help="Custom title for the viewer HTML page")
    p.add_argument("--no_showcase", type=int, default=0,
                   help="Skip showcase video rendering (1=yes, 0=no)")
    # Batch
    p.add_argument("--batch_list", type=str, default="",
                   help="Text file with list of input directories (one per line)")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load config file if provided
    cfg_overrides = {}
    if args.config:
        with open(args.config) as f:
            cfg_overrides = json.load(f)

    # Build pipeline config
    cfg = PipelineConfig(
        input_dir=args.input_dir,
        work_dir=args.work_dir,
        sr_mode=args.sr_mode,
        sr_model=args.sr_model,
        sr_scale=args.sr_scale,
        sr_device=args.sr_device,
        sr_model_load_timeout_s=args.sr_model_load_timeout,
        sr_frame_timeout_s=args.sr_frame_timeout,
        sr_strict_model=args.sr_strict_model,
        colmap_path=args.colmap_path,
        colmap_camera_model=args.colmap_camera,
        colmap_camera_fallbacks=tuple(
            v.strip() for v in args.colmap_camera_fallbacks.split(",") if v.strip()
        ),
        colmap_min_registered_ratio=args.colmap_min_registered_ratio,
        colmap_min_registered_images=args.colmap_min_registered_images,
        colmap_gpu=args.colmap_gpu,
        train_max_steps=args.max_steps,
        train_warmup_steps=args.warmup_steps,
        train_device=args.train_device,
        skip_existing=bool(args.skip_existing),
        render_viewer=not bool(args.no_viewer),
        viewer_title=args.viewer_title,
        render_showcase=not bool(args.no_showcase),
    )

    # Apply config file overrides
    for k, v in cfg_overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Batch mode
    if args.batch_list:
        return run_batch(args.batch_list, cfg, args)

    # Single run
    if not cfg.input_dir:
        parser.error("--input_dir is required (or use --batch_list)")

    pipeline = Pipeline(cfg)
    pipeline.run(start_step=args.start, end_step=args.end)
    pipeline.export_config()


def run_batch(batch_file: str, base_cfg: PipelineConfig, args):
    """Process multiple scenes from a batch list file."""
    with open(batch_file) as f:
        scenes = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Batch processing {len(scenes)} scenes:")
    for s in scenes:
        print(f"  - {s}")

    failed = []
    for i, scene_path in enumerate(scenes):
        print(f"\n{'#' * 60}")
        print(f"  BATCH [{i+1}/{len(scenes)}]: {scene_path}")
        print(f"{'#' * 60}")

        scene_name = os.path.basename(scene_path.rstrip("/\\"))
        cfg = PipelineConfig(**{
            **base_cfg.__dict__,
            "input_dir": scene_path,
            "work_dir": os.path.join(base_cfg.work_dir, scene_name),
        })

        try:
            pipeline = Pipeline(cfg)
            pipeline.run(start_step=args.start, end_step=args.end)
        except Exception as e:
            print(f"[BATCH ERROR] {scene_path}: {e}")
            failed.append((scene_path, str(e)))

    print(f"\n{'=' * 60}")
    print(f"  Batch complete. {len(scenes) - len(failed)}/{len(scenes)} succeeded.")
    if failed:
        print(f"  Failed scenes:")
        for path, err in failed:
            print(f"    - {path}: {err}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
