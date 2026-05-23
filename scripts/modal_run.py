#!/usr/bin/env python3
"""Serverless SR-3DGS via Modal — pay only for GPU seconds used.

Setup (one time):
    pip install modal
    modal token new          # Authenticate

Usage:
    # Run on Modal cloud GPU (serverless)
    python scripts/modal_run.py --video client_video.mp4 --preset standard

    # Deploy as a persistent API service
    python scripts/modal_run.py --deploy

    # Check status
    modal app list
    modal volume ls sr3dgs-outputs

Environment persistence:
    - Docker image: cached on Modal's registry (no rebuild after first deploy)
    - Outputs: stored on Modal Volume sr3dgs-outputs (persists between runs)
    - You don't manage any servers — Modal handles everything
"""

import os
import sys
import argparse

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _parent_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Serverless SR-3DGS via Modal"
    )
    parser.add_argument("--video", type=str, default="",
                        help="Video URL or path (local path auto-uploaded)")
    parser.add_argument("--preset", type=str, default="standard",
                        choices=["fast", "standard", "quality", "autodl", "extreme"])
    parser.add_argument("--work_subdir", type=str, default="",
                        help="Subdirectory on volume for this job")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy as persistent API service")
    parser.add_argument("--gpu", type=str, default="A10G",
                        choices=["A10G", "A100", "A100-80GB"],
                        help="GPU type for Modal")
    parser.add_argument("--notify", type=str, default="",
                        help="Webhook URL for completion notification")

    args = parser.parse_args()

    # Check modal
    try:
        import modal
    except ImportError:
        print("Modal SDK not installed. Run: pip install modal")
        print("Then: modal token new  (one-time authentication)")
        sys.exit(1)

    if args.deploy:
        from sr_3dgs.modal_pipeline import build_and_deploy
        app = build_and_deploy("sr-3dgs")
        print(f"\nDeploying to Modal...")
        print(f"After deploy, call via REST API:")
        print(f"  POST https://<username>--sr-3dgs-process-video.modal.run")
        return

    if not args.video:
        parser.error("--video is required (or use --deploy)")

    from sr_3dgs.modal_pipeline import run_pipeline_on_modal

    app = run_pipeline_on_modal(
        video_source=args.video,
        preset=args.preset,
        work_subdir=args.work_subdir,
        notify_webhook=args.notify,
    )


if __name__ == "__main__":
    main()
