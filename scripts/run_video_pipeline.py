#!/usr/bin/env python3
"""Video → 3DGS Pipeline CLI — turn client videos into 3D deliverables.

Usage:
    # Basic: video to 3D viewer + showcase
    python run_video_pipeline.py --video client_video.mp4

    # High quality (slower)
    python run_video_pipeline.py --video client_video.mp4 --preset quality

    # Fast preview (for client demos)
    python run_video_pipeline.py --video client_video.mp4 --preset fast

    # Custom settings
    python run_video_pipeline.py --video input.mp4 \
        --extract_fps 5 --sr_model "real-esrgan" --sr_scale 4 \
        --train_steps 20000 --output_name "client_scene"

Output (in workspace_video/<name>/):
    <name>.splat           — Compressed 3D model (~5-30 MB)
    <name>_viewer.html     — Self-contained mobile viewer
    <name>_showcase.mp4    — Trajectory video for portfolio
    clean_output/*.ply     — Clean PLY files
"""

import os
import sys
import argparse

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from sr_3dgs.video_pipeline import VideoPipeline, VideoPipelineConfig


# ── Presets ──
PRESETS = {
    "debug": {
        "extract_fps": 8.0,
        "extract_min_sharpness": 30.0,
        "extract_min_frame_diff": 0.05,
        "extract_max_frames": 30,
        "sr_model": "real-esrgan",
        "sr_scale": 1,
        "train_max_steps": 1000,
        "train_warmup_steps": 200,
        "train_cap_max": 50_000,
        "max_web_gaussians": 50_000,
        "showcase_duration": 3.0,
    },
    "fast": {
        "extract_fps": 5.0,
        "extract_min_sharpness": 50.0,
        "extract_min_frame_diff": 0.03,
        "extract_max_frames": 100,
        "sr_model": "real-esrgan",
        "sr_scale": 1,
        "train_max_steps": 8_000,
        "train_warmup_steps": 300,
        "train_cap_max": 100_000,
        "max_web_gaussians": 100_000,
        "showcase_duration": 5.0,
    },
    "standard": {
        "extract_fps": 3.0,
        "extract_min_sharpness": 80.0,
        "extract_min_frame_diff": 0.02,
        "extract_max_frames": 200,
        "sr_model": "real-esrgan",
        "sr_scale": 1,
        "train_max_steps": 15_000,
        "train_warmup_steps": 500,
        "train_cap_max": 200_000,
        "max_web_gaussians": 200_000,
        "showcase_duration": 8.0,
    },
    "quality": {
        "extract_fps": 2.0,
        "extract_min_sharpness": 100.0,
        "extract_min_frame_diff": 0.01,
        "extract_max_frames": 300,
        "sr_model": "real-esrgan",
        "sr_scale": 2,
        "train_max_steps": 20_000,
        "train_warmup_steps": 800,
        "train_cap_max": 300_000,
        "max_web_gaussians": 300_000,
        "showcase_duration": 10.0,
    },
    "extreme": {
        "extract_fps": 2.0,
        "extract_min_sharpness": 30.0,
        "extract_min_frame_diff": 0.02,
        "extract_max_frames": 250,
        "sr_model": "supir",
        "sr_scale": 4,
        "sr_kwargs": {
            "guidance_scale": 7.5,
            "num_steps": 50,
        },
        "train_max_steps": 25_000,
        "train_warmup_steps": 1_000,
        "train_cap_max": 200_000,
        "max_web_gaussians": 200_000,
        "showcase_duration": 8.0,
    },
    "autodl": {
        "extract_fps": 2.0,
        "extract_min_sharpness": 80.0,
        "extract_min_frame_diff": 0.015,
        "extract_max_frames": 350,
        "sr_model": "real-esrgan",
        "sr_scale": 2,
        "train_max_steps": 25_000,
        "train_warmup_steps": 800,
        "train_cap_max": 400_000,
        "max_web_gaussians": 400_000,
        "showcase_duration": 10.0,
    },
}
def main():
    parser = argparse.ArgumentParser(
        description="Video → 3DGS: Turn client videos into 3D deliverables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output_name", type=str, default="",
                        help="Name for output files (default: video filename)")
    parser.add_argument("--work_dir", type=str, default="workspace_video",
                        help="Root working directory")
    parser.add_argument("--final_output_dir", type=str, default="output",
                        help="Root final deliverables directory")
    parser.add_argument("--preset", type=str, default="standard",
                        choices=list(PRESETS.keys()),
                        help="Configuration preset")
    parser.add_argument("--projection", type=str, default="auto",
                        choices=["auto", "perspective", "equirectangular"],
                        help="Input video projection. Auto treats ~2:1 video as 360 equirectangular.")

    # Frame extraction overrides
    parser.add_argument("--extract_fps", type=float, default=None)
    parser.add_argument("--extract_max_frames", type=int, default=None)
    parser.add_argument("--start_time", type=float, default=None,
                        help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration in seconds")
    parser.add_argument("--equirect_face_size", type=int, default=1024,
                        help="Cube-face size when --projection equirectangular/auto detects 360")
    parser.add_argument("--equirect_faces", default="front,right,back,left",
                        help="Comma-separated cube faces for 360 extraction")

    # SR overrides
    parser.add_argument("--sr_model", type=str, default=None,
                        choices=["real-esrgan", "dat", "supir", "basicvsr++"])
    parser.add_argument("--sr_scale", type=int, default=None,
                        choices=[2, 4, 8])

    # Training overrides
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--train_data_factor", type=int, default=None,
                        help="Downscale factor for training renders (2 = half res, saves VRAM)")
    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument("--object_bbox", type=str, default="",
                        help="Optional object crop after alignment: left,top,right,bottom")
    parser.add_argument("--object_mask", choices=["off", "auto"], default="off",
                        help="Build object masks and use mask-aware training")
    parser.add_argument("--object_mask_rect", type=str, default="",
                        help="Optional mask foreground prior x0,y0,x1,y1 after crop/alignment")
    parser.add_argument("--object_mask_method", choices=["fast", "grabcut", "rembg"], default="fast",
                        help="Mask generation backend")
    parser.add_argument("--mask_background_weight", type=float, default=0.05,
                        help="Loss weight outside foreground mask")
    parser.add_argument("--mask_alpha_reg", type=float, default=0.05,
                        help="Alpha penalty outside foreground mask")
    parser.add_argument("--cluster_clean", action="store_true",
                        help="Run voxel connected-component cleanup before web/pro PLY export")
    parser.add_argument("--no_cluster_auto", action="store_true",
                        help="Disable automatic cluster-clean seed/radius/scale estimation")
    parser.add_argument("--cluster_preset", choices=["conservative", "balanced", "aggressive"],
                        default="balanced", help="Cleanup preset")
    parser.add_argument("--cluster_opacity_min", type=float, default=0.08)
    parser.add_argument("--cluster_voxel_size", type=float, default=0.10)
    parser.add_argument("--cluster_max_radius", type=float, default=0.0,
                        help="Optional radius around --cluster_seed; 0 disables")
    parser.add_argument("--cluster_max_scale", type=float, default=0.0,
                        help="Optional max actual Gaussian scale; 0 disables")
    parser.add_argument("--cluster_select", choices=["nearest", "largest"], default="nearest")
    parser.add_argument("--cluster_seed", default="0,0,0",
                        help="Object seed point for nearest-component cleanup")
    parser.add_argument("--cluster_min_component_points", type=int, default=1000)
    parser.add_argument("--cluster_dilate", type=int, default=1)

    # Output options
    parser.add_argument("--no_showcase", action="store_true",
                        help="Skip showcase video rendering")
    parser.add_argument("--no_embed_splat", action="store_true",
                        help="Don't embed .splat in HTML (smaller HTML, needs server)")
    parser.add_argument("--no_sog", action="store_true",
                        help="Skip PlayCanvas SOG export")
    parser.add_argument("--no_sog_viewer", action="store_true",
                        help="Skip official PlayCanvas SOG HTML viewer")
    parser.add_argument("--no_publish_output", action="store_true",
                        help="Do not publish flat final files to output/<name>")
    parser.add_argument("--no_assess_inputs", action="store_true",
                        help="Skip lightweight frame/mask input quality reports")
    parser.add_argument("--viewer_title", type=str, default="",
                        help="Custom viewer title")

    # Batch
    parser.add_argument("--batch", type=str, default="",
                        help="Text file with list of video paths (one per line)")

    args = parser.parse_args()

    if args.batch:
        return run_batch(args)

    run_single(args)


def run_single(args):
    """Process a single video."""
    preset = PRESETS[args.preset].copy()

    # Build config from preset + CLI overrides
    cfg = VideoPipelineConfig(
        video_path=args.video,
        work_dir=args.work_dir,
        final_output_dir=args.final_output_dir,
        output_name=args.output_name,
        projection=args.projection,
        extract_fps=args.extract_fps or preset.get("extract_fps", 3.0),
        extract_min_sharpness=preset.get("extract_min_sharpness", 80.0),
        extract_min_frame_diff=preset.get("extract_min_frame_diff", 0.02),
        extract_max_frames=args.extract_max_frames or preset.get("extract_max_frames", 200),
        equirect_face_size=args.equirect_face_size,
        equirect_faces=tuple(v.strip() for v in args.equirect_faces.split(",") if v.strip()),
        start_time=args.start_time,
        duration=args.duration,
        sr_model=args.sr_model or preset.get("sr_model", "real-esrgan"),
        sr_scale=args.sr_scale or preset.get("sr_scale", 4),
        sr_kwargs=preset.get("sr_kwargs", {}),
        train_max_steps=args.train_steps or preset.get("train_max_steps", 25_000),
        train_warmup_steps=preset.get("train_warmup_steps", 800),
        train_data_factor=args.train_data_factor or preset.get("train_data_factor", 1),
        train_cap_max=preset.get("train_cap_max", 200_000),
        train_device=args.train_device,
        object_bbox=args.object_bbox,
        object_mask=args.object_mask,
        object_mask_rect=args.object_mask_rect,
        object_mask_method=args.object_mask_method,
        mask_background_weight=args.mask_background_weight,
        mask_alpha_reg=args.mask_alpha_reg,
        cluster_clean=args.cluster_clean,
        cluster_auto=not args.no_cluster_auto,
        cluster_preset=args.cluster_preset,
        cluster_opacity_min=args.cluster_opacity_min,
        cluster_voxel_size=args.cluster_voxel_size,
        cluster_max_radius=args.cluster_max_radius,
        cluster_max_scale=args.cluster_max_scale,
        cluster_select=args.cluster_select,
        cluster_seed=args.cluster_seed,
        cluster_min_component_points=args.cluster_min_component_points,
        cluster_dilate=args.cluster_dilate,
        max_web_gaussians=preset.get("max_web_gaussians", 2_000_000),
        viewer_title=args.viewer_title,
        viewer_embed_splat=not args.no_embed_splat,
        export_sog=not args.no_sog,
        generate_sog_viewer=not args.no_sog_viewer,
        publish_output=not args.no_publish_output,
        assess_inputs=not args.no_assess_inputs,
        render_showcase=not args.no_showcase,
        showcase_duration=preset.get("showcase_duration", 8.0),
    )

    pipeline = VideoPipeline(cfg)
    results = pipeline.run()

    # Print client-ready summary
    print(f"\n{'─'*50}")
    print(f"  READY FOR DELIVERY")
    print(f"{'─'*50}")
    if "viewer_html" in results:
        print(f"  Viewer:  {results['viewer_html']}")
        print(f"    ↑ Send this HTML to client (works on mobile)")
    if "showcase_video" in results:
        print(f"  Video:   {results['showcase_video']}")
    if "splat_file" in results:
        print(f"  Splat:   {results['splat_file']}")
    if "standard_ply" in results:
        print(f"  Std PLY: {results['standard_ply']}")
    if "sog_file" in results:
        print(f"  SOG:     {results['sog_file']}")
    if "sog_viewer_html" in results:
        print(f"  SOG UI:  {results['sog_viewer_html']}")
    if "delivery" in results:
        print(f"  Delivery:{results['delivery']}")
    if "final_output" in results:
        print(f"  Output:  {results['final_output']}")
    if "clean_ply" in results:
        print(f"  PLY:     {results.get('clean_ply', 'N/A')}")
    print(f"{'─'*50}")


def run_batch(args):
    """Batch process multiple videos."""
    with open(args.batch) as f:
        videos = [line.strip() for line in f
                  if line.strip() and not line.startswith("#")]

    if not videos:
        print("No videos found in batch file.")
        return

    print(f"Batch processing {len(videos)} videos:")
    for v in videos:
        print(f"  - {v}")

    for i, video_path in enumerate(videos):
        print(f"\n{'#'*60}")
        print(f"  BATCH [{i+1}/{len(videos)}]: {video_path}")
        print(f"{'#'*60}")
        try:
            # Override video_path in args and run single
            import copy
            batch_args = copy.copy(args)
            batch_args.video = video_path
            batch_args.batch = ""  # prevent recursion
            run_single(batch_args)
        except Exception as e:
            print(f"[BATCH ERROR] {video_path}: {e}")


if __name__ == "__main__":
    main()
