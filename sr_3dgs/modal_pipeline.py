"""Modal serverless GPU pipeline for SR-3DGS.

Uses Modal (modal.com) to run the entire pipeline on cloud GPUs
without managing any servers. Environment is cached as a Docker image,
outputs persist on a Modal Volume.

First time: `modal deploy`  →  builds image (5-8 min, cached forever)
Every time:  `modal run`     →  reuses cached image, runs pipeline

Usage:
    modal run scripts/modal_run.py --video s3://bucket/video.mp4 --preset autodl
    modal run scripts/modal_run.py --video https://example.com/video.mp4 --preset standard

Pricing: A10G ~$1.10/h, A100-40GB ~$3.50/h, A100-80GB ~$5.00/h
"""

import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# Modal is imported at function level to avoid requiring it locally


def _check_modal():
    """Verify modal CLI is installed and authenticated."""
    try:
        import modal
    except ImportError:
        print("Modal not installed. Run: pip install modal && modal setup")
        sys.exit(1)

    # Check if token is set
    try:
        result = subprocess.run(
            ["modal", "config", "show"], capture_output=True, text=True
        )
        if "not authenticated" in result.stderr.lower():
            print("Not authenticated. Run: modal token new")
            sys.exit(1)
    except Exception:
        pass


def create_modal_image(python_version: str = "3.10") -> "modal.Image":
    """Build the Docker image with all SR-3DGS dependencies."""
    import modal

    image = (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install("colmap", "ffmpeg", "libsm6", "libxext6", "libxrender-dev")
        .pip_install(
            "torch", "torchvision",
            "numpy<2.0.0", "Pillow", "scikit-learn",
            "opencv-python-headless", "imageio[ffmpeg]",
            "torchmetrics[image]", "tqdm",
            "gsplat",
            "realesrgan", "basicsr",
        )
        .pip_install(".")  # Install sr_3dgs itself
    )
    return image


def get_or_create_volume(name: str = "sr3dgs-outputs") -> "modal.Volume":
    """Get or create a persistent volume for outputs."""
    import modal
    try:
        return modal.Volume.lookup(name)
    except modal.exception.NotFoundError:
        return modal.Volume.from_name(name, create_if_missing=True)


def run_pipeline_on_modal(video_source: str,
                          preset: str = "standard",
                          work_subdir: str = "",
                          result_format: str = "all",
                          notify_webhook: str = ""):
    """Run the full video pipeline on Modal and return output URLs.

    Args:
        video_source: URL or S3 path to the input video
        preset: 'fast', 'standard', 'quality', 'autodl', 'extreme'
        work_subdir: Subdirectory on volume for this job
        result_format: 'all', 'viewer', 'video', 'splat', 'ply'
        notify_webhook: POST results URL on completion

    Returns:
        Dict with paths/URLs to all outputs on the volume
    """
    import modal

    _check_modal()

    image = create_modal_image()
    volume = get_or_create_volume()

    app = modal.App("sr-3dgs-pipeline", image=image)

    @app.function(
        gpu=modal.gpu.A10G(count=1),
        volumes={"/outputs": volume},
        timeout=3600 * 3,  # 3 hour max
        cpu=8,
        memory=32768,
    )
    def train(video_path: str, preset_name: str, subdir: str):
        """Run the training pipeline on Modal GPU."""
        import subprocess
        import sys

        output_dir = f"/outputs/{subdir}" if subdir else "/outputs"

        cmd = [
            sys.executable, "-m", "sr_3dgs.scripts.run_video_pipeline",
            "--video", video_path,
            "--preset", preset_name,
            "--work_dir", output_dir,
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"Pipeline failed with code {result.returncode}")

        # List outputs
        outputs = {}
        for ext, key in [("_viewer.html", "viewer"), ("_showcase.mp4", "video"),
                          (".splat", "splat"), (".ply", "ply")]:
            matches = list(Path(output_dir).rglob(f"*{ext}"))
            if matches:
                outputs[key] = str(matches[0])
        return outputs

    @app.local_entrypoint()
    def main():
        print(f"[Modal] Launching SR-3DGS pipeline")
        print(f"  Video: {video_source}")
        print(f"  Preset: {preset}")
        print(f"  Volume: /outputs/{work_subdir or '(root)'}")

        t0 = time.time()
        result = train.remote(video_source, preset, work_subdir)
        elapsed = time.time() - t0

        print(f"\n[Modal] Complete in {elapsed:.0f}s")
        for k, v in result.items():
            print(f"  {k}: {v}")

        if notify_webhook:
            import requests
            requests.post(notify_webhook, json=result)

        return result

    return app


def build_and_deploy(app_name: str = "sr-3dgs"):
    """Build the Modal app and deploy it (one-time setup).

    After deployment, the Docker image is cached on Modal's registry.
    Subsequent calls reuse the cached image (no rebuild).
    """
    import modal

    image = create_modal_image()
    volume = get_or_create_volume()

    app = modal.App(app_name, image=image)

    @app.cls(
        gpu=modal.gpu.A10G(count=1),
        volumes={"/outputs": volume},
        timeout=3600 * 3,
        cpu=8,
        memory=32768,
        allow_concurrent_inputs=1,
    )
    class SR3DGSPipeline:
        """Modal-deployed SR-3DGS pipeline as a service."""

        @modal.enter()
        def setup(self):
            """Called once per container start — GPU warmup."""
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Modal Worker] GPU: {torch.cuda.get_device_name(0)}")

        @modal.method()
        def process_video(self, video_data: bytes, filename: str = "input.mp4",
                          preset: str = "standard") -> dict:
            """Process a video (received as bytes) and return results."""
            import subprocess

            # Write video to temp
            video_path = f"/tmp/{filename}"
            with open(video_path, "wb") as f:
                f.write(video_data)

            job_id = filename.rsplit(".", 1)[0]
            output_dir = f"/outputs/{job_id}"

            result = subprocess.run([
                "python", "-m", "sr_3dgs.scripts.run_video_pipeline",
                "--video", video_path,
                "--preset", preset,
                "--work_dir", output_dir,
            ], capture_output=True, text=True, timeout=3600 * 2)

            return {
                "job_id": job_id,
                "success": result.returncode == 0,
                "stdout": result.stdout[-2000:],
                "output_dir": output_dir,
            }

        @modal.method()
        def get_result(self, job_id: str) -> Optional[dict]:
            """Check if a job's results are ready."""
            import glob
            output_dir = f"/outputs/{job_id}"
            files = glob.glob(f"{output_dir}/**/*", recursive=True)
            return {
                "job_id": job_id,
                "files": files,
            }

    print(f"[Modal] Deploying '{app_name}'...")
    print(f"  Image: cached after first deploy (5-8 min first time, <10s after)")
    print(f"  Volume: sr3dgs-outputs (persistent)")
    print(f"  GPU: A10G (24GB)")
    return app
