#!/usr/bin/env python3
"""Pack up pipeline outputs into a clean client delivery folder.

Usage:
    python scripts/package_delivery.py \
        --work_dir workspace_video/client_scene \
        --output delivery_张先生 \
        --viewer_title "张先生别墅 - 3D展示"
"""

import os, sys, shutil, argparse
from pathlib import Path

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _parent_dir)


def main():
    parser = argparse.ArgumentParser(description="Package delivery for client")
    parser.add_argument("--work_dir", required=True, help="Pipeline output directory")
    parser.add_argument("--output", required=True, help="Delivery folder name")
    parser.add_argument("--viewer_title", default="3D Scene", help="Title for HTML viewer")
    parser.add_argument("--include_ply", action="store_true", default=True, help="Include PLY files")
    parser.add_argument("--include_checkpoint", action="store_true", default=False)
    parser.add_argument("--no_viewer", action="store_true", help="Skip HTML viewer generation")

    args = parser.parse_args()
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output)

    if not work_dir.exists():
        print(f"ERROR: {work_dir} not found")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    files_copied = []

    # 1. Copy HTML viewer
    viewer_candidates = sorted(work_dir.glob("*_viewer.html"))
    if viewer_candidates:
        for v in viewer_candidates:
            dst = output_dir / "3D_viewer.html"
            shutil.copy2(v, dst)
            size = dst.stat().st_size / (1024*1024)
            files_copied.append(f"3D_viewer.html ({size:.1f} MB)")
            print(f"[OK] Viewer: {dst}")

    # 2. Copy showcase video
    video_candidates = sorted(work_dir.glob("*_showcase.mp4"))
    if video_candidates:
        for v in video_candidates:
            dst = output_dir / "showcase.mp4"
            shutil.copy2(v, dst)
            size = dst.stat().st_size / (1024*1024)
            files_copied.append(f"showcase.mp4 ({size:.1f} MB)")
            print(f"[OK] Video: {dst}")

    # 3. Copy .splat
    splat_files = sorted(work_dir.glob("*.splat"))
    if splat_files:
        dst = output_dir / "model.splat"
        shutil.copy2(splat_files[0], dst)
        size = dst.stat().st_size / (1024*1024)
        files_copied.append(f"model.splat ({size:.1f} MB)")
        print(f"[OK] Splat: {dst}")

    # 4. Copy clean PLY
    if args.include_ply:
        ply_files = sorted(work_dir.glob("clean_output/*.ply"))
        if ply_files:
            dst_dir = output_dir / "ply_models"
            dst_dir.mkdir(exist_ok=True)
            for ply in ply_files:
                dst = dst_dir / ply.name
                shutil.copy2(ply, dst)
                size = dst.stat().st_size / (1024*1024)
                files_copied.append(f"ply_models/{ply.name} ({size:.1f} MB)")
            print(f"[OK] PLY files: {len(ply_files)} files")

    # 5. Copy checkpoint (optional)
    if args.include_checkpoint:
        ckpt_files = sorted(work_dir.glob("train_output/checkpoint_step*.pt"))
        if ckpt_files:
            dst = output_dir / "model_checkpoint.pt"
            shutil.copy2(ckpt_files[-1], dst)
            size = dst.stat().st_size / (1024*1024)
            files_copied.append(f"model_checkpoint.pt ({size:.1f} MB)")
            print(f"[OK] Checkpoint: {dst}")

    # 6. Generate viewer from .splat if not already in output
    if not args.no_viewer and splat_files and not viewer_candidates:
        from sr_3dgs.web_viewer import generate_viewer
        viewer_out = output_dir / "3D_viewer.html"
        generate_viewer(str(splat_files[0]), str(viewer_out), title=args.viewer_title)
        files_copied.append("3D_viewer.html (generated)")

    # 7. Summary
    print(f"\n{'='*50}")
    print(f"  DELIVERY PACKAGE: {output_dir}")
    print(f"  Files:")
    for f in files_copied:
        print(f"    {f}")
    print(f"{'='*50}")
    print(f"\n  Send this folder to client (zip it first)")
    print(f"  Client opens 3D_viewer.html in browser (mobile OK)")

    # 8. Create delivery README
    readme = output_dir / "README.txt"
    readme.write_text(f"""3D Scene Delivery
=================

Open '3D_viewer.html' in any browser (Chrome, Safari, mobile browsers).
- Touch: drag to rotate, pinch to zoom
- Mouse: click+drag to rotate, scroll to zoom
- Press 'R' or tap reset button to return to default view

Files:
- 3D_viewer.html: Self-contained 3D viewer (mobile-ready)
- showcase.mp4: Camera flythrough video
- model.splat: Compressed 3D model
- ply_models/: Standard PLY files (import into Blender, Unity, etc.)

Need help? Contact the provider.
""")
    print(f"[OK] README: {readme}")


if __name__ == "__main__":
    main()