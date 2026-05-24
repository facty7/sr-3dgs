"""Main pipeline orchestrator for SR + 3DGS.

The Pipeline class executes the full decoupled SR-3DGS workflow:

    Step 1: COLMAP on ORIGINAL images (几何锚定)
    Step 2: Super-Resolution (独立超分)
    Step 3: Intrinsic Alignment (内参数学对齐)
    Step 4: 3DGS Training (注入高分数据训练)
    Step 5: Cleanup & Export (降噪修剪)

Each step can be run independently or as part of the full pipeline.
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .step1_colmap import COLMAPExtractor
from .step2_super_resolution import SuperResolutionProcessor
from .step3_intrinsic_align import IntrinsicAligner
from .step4_train_3dgs import SR3DGSTrainer, SRTrainConfig
from .step5_cleanup import CleanupProcessor
from .utils import ensure_dir, get_gpu_memory, vram_safe_config, check_dependencies, print_dep_check


@dataclass
class PipelineConfig:
    """Global configuration for the SR + 3DGS pipeline."""

    # Input/Output
    input_dir: str = ""                          # Raw input images or video frames
    work_dir: str = "workspace"                  # Working directory for all outputs

    # Step 1: COLMAP
    colmap_camera_model: str = "SIMPLE_PINHOLE"
    colmap_path: str = "colmap"
    colmap_gpu: int = 0

    # Step 2: Super Resolution
    sr_mode: str = "auto"                       # auto | off | resize | model
    sr_model: str = "real-esrgan"                # real-esrgan | dat | supir | basicvsr++
    sr_scale: int = 4
    sr_device: str = "cuda"
    sr_kwargs: Dict[str, Any] = field(default_factory=dict)
    sr_model_load_timeout_s: int = 180
    sr_frame_timeout_s: int = 300
    sr_strict_model: bool = False

    # Step 3: Intrinsic alignment (uses sr_scale automatically)

    # Step 4: Training
    train_max_steps: int = 15_000
    train_warmup_steps: int = 500
    train_data_factor: int = 1
    train_max_render_dim: int = 1600  # Cap max render dimension during training
    train_device: str = "cuda"
    train_sh_degree: int = 2
    train_strategy: str = "default"
    train_cap_max: int = 200_000  # MCMC hard cap when train_strategy="mcmc"
    train_mask_dir: str = ""
    train_mask_background_weight: float = 0.05
    train_mask_alpha_reg: float = 0.05

    # Step 5: Cleanup
    cleanup_opacity_thresholds: tuple = (0.05, 0.1, 0.2, 0.3)

    # Step 6: Export (viewer + splat + showcase)
    render_viewer: bool = True                   # Generate HTML viewer
    viewer_title: str = ""                       # Viewer page title (default: scene name)
    render_showcase: bool = True                 # Generate showcase video
    showcase_duration: float = 8.0               # Seconds
    showcase_trajectory: str = "spiral"
    max_web_gaussians: int = 2_000_000           # Cap for mobile performance
    viewer_embed_splat: bool = True              # Embed splat in HTML

    # Execution
    skip_existing: bool = True                   # Skip steps with existing output
    stop_on_error: bool = True
    verbose: bool = True


class Pipeline:
    """End-to-end SR + 3DGS pipeline.

    Usage:
        config = PipelineConfig(input_dir="/path/to/images")
        pipeline = Pipeline(config)
        pipeline.run()

    Or step by step:
        pipeline.run_step1()
        pipeline.run_step2()
        pipeline.run_step3()
        pipeline.run_step4()
        pipeline.run_step5()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._setup_dirs()
        self._step_results: Dict[str, Any] = {}
        self._check_vram()

    def _check_vram(self):
        """Print VRAM info and warnings."""
        mem = get_gpu_memory()
        if mem["total_gb"] > 0:
            print(f"[Pipeline] GPU: {mem['total_gb']:.1f} GB VRAM, "
                  f"{mem['free_gb']:.1f} GB free")
            if mem["total_gb"] <= 8:
                safe = vram_safe_config(mem["total_gb"], 3840, 2160)
                for w in safe.get("warnings", []):
                    print(f"[Pipeline] VRAM WARNING: {w}")

    def _setup_dirs(self):
        cfg = self.config
        self.work_dir = Path(cfg.work_dir)
        self.colmap_dir = self.work_dir / "colmap"
        self.sr_dir = self.work_dir / "sr_images"
        self.aligned_dir = self.work_dir / "aligned"
        self.train_dir = self.work_dir / "train_output"
        self.clean_dir = self.work_dir / "clean_output"

    def run(self, start_step: int = 1, end_step: int = 6,
            skip_dep_check: bool = False):
        """Run the full pipeline from start_step to end_step (inclusive)."""
        cfg = self.config

        if not skip_dep_check and start_step <= 1:
            status = check_dependencies()
            print_dep_check(status)
            if not status["ok"]:
                raise RuntimeError(
                    "Missing required dependencies. "
                    "Run: bash scripts/setup_local.sh"
                )

        print("=" * 60)
        print("  SR + 3DGS Pipeline")
        print(f"  Input: {cfg.input_dir}")
        print(f"  SR: {cfg.sr_mode} / {cfg.sr_model} (x{cfg.sr_scale})")
        print(f"  Work Dir: {cfg.work_dir}")
        print("=" * 60)

        t0 = time.time()

        steps = {
            1: self.run_step1,
            2: self.run_step2,
            3: self.run_step3,
            4: self.run_step4,
            5: self.run_step5,
            6: self.run_step6,
        }

        for step_num in range(start_step, end_step + 1):
            print(f"\n{'─' * 40}")
            print(f"  STEP {step_num}")
            print(f"{'─' * 40}")
            try:
                steps[step_num]()
            except Exception as e:
                print(f"[ERROR] Step {step_num} failed: {e}")
                if cfg.stop_on_error:
                    raise
                print("[WARNING] Continuing despite error (stop_on_error=False)")

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"  Pipeline complete in {elapsed:.0f}s")
        print(f"  Results: {self.work_dir}")
        self._print_deliverables()
        print(f"{'=' * 60}")

    def run_step1(self):
        """Step 1: Run COLMAP on original (low-res) images."""
        cfg = self.config
        extractor = COLMAPExtractor(
            image_dir=cfg.input_dir,
            work_dir=self.colmap_dir,
            camera_model=cfg.colmap_camera_model,
            colmap_path=cfg.colmap_path,
            gpu_index=cfg.colmap_gpu,
        )
        self._step_results["colmap_sparse"] = extractor.run(
            force=not cfg.skip_existing
        )

    def run_step2(self):
        """Step 2: Super-resolve all images."""
        cfg = self.config
        processor = SuperResolutionProcessor(
            image_dir=cfg.input_dir,
            output_dir=self.sr_dir,
            sr_model_name=cfg.sr_model,
            scale=cfg.sr_scale,
            device=cfg.sr_device,
            model_kwargs=cfg.sr_kwargs,
            mode=cfg.sr_mode,
            model_load_timeout_s=cfg.sr_model_load_timeout_s,
            frame_timeout_s=cfg.sr_frame_timeout_s,
            strict_model=cfg.sr_strict_model,
        )
        self._step_results["sr_images"] = processor.run(
            force=not cfg.skip_existing
        )
        if processor.manifest_path.exists():
            self._step_results["sr_manifest"] = str(processor.manifest_path)
        processor.cleanup()

    def run_step3(self):
        """Step 3: Scale camera intrinsics and align data."""
        cfg = self.config
        sparse_dir = self._step_results.get("colmap_sparse",
                                             self.colmap_dir / "sparse" / "0")

        aligner = IntrinsicAligner(
            colmap_sparse_dir=str(sparse_dir),
            sr_image_dir=str(self.sr_dir),
            output_dir=str(self.aligned_dir),
            scale_factor=cfg.sr_scale,
        )
        self._step_results["aligned"] = aligner.run(
            force=not cfg.skip_existing
        )

    def run_step4(self):
        """Step 4: Train 3DGS with SR-optimized strategy."""
        cfg = self.config
        train_cfg = SRTrainConfig(
            data_dir=str(self.aligned_dir),
            result_dir=str(self.train_dir),
            max_steps=cfg.train_max_steps,
            warmup_steps=cfg.train_warmup_steps,
            sh_degree=cfg.train_sh_degree,
            device=cfg.train_device,
            data_factor=cfg.train_data_factor,
            max_render_dim=cfg.train_max_render_dim,
            strategy=cfg.train_strategy,
            cap_max=cfg.train_cap_max,
            mask_dir=cfg.train_mask_dir,
            mask_background_weight=cfg.train_mask_background_weight,
            mask_alpha_reg=cfg.train_mask_alpha_reg,
        )
        trainer = SR3DGSTrainer(train_cfg)
        trainer.run()
        # Find latest checkpoint
        ckpts = sorted(self.train_dir.glob("checkpoint_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
        if ckpts:
            self._step_results["checkpoint"] = str(ckpts[-1])
        summary = self.train_dir / "training_summary.json"
        if summary.exists():
            self._step_results["training_summary"] = str(summary)

    def run_step5(self):
        """Step 5: Cleanup floaters and export clean PLY."""
        cfg = self.config
        ckpt = self._step_results.get("checkpoint")
        if not ckpt:
            ckpts = sorted(self.train_dir.glob("checkpoint_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
            if ckpts:
                ckpt = str(ckpts[-1])  # Use latest checkpoint
                print(f"[Step5] Using latest checkpoint: {Path(ckpt).name}")
            else:
                ckpt = None

        if not ckpt or not Path(ckpt).exists():
            raise FileNotFoundError(
                f"No checkpoint found for cleanup. Run step 4 first."
            )

        cleaner = CleanupProcessor(
            checkpoint_path=ckpt,
            output_dir=str(self.clean_dir),
        )
        cleaner.run(opacity_thresholds=cfg.cleanup_opacity_thresholds)
        self._step_results["clean"] = str(self.clean_dir)

    def run_step6(self):
        cfg = self.config
        scene_name = Path(cfg.work_dir).name

        clean_ply = None
        if self.clean_dir.exists():
            ply_files = sorted(self.clean_dir.glob("*.ply"))
            if ply_files:
                for ply in ply_files:
                    if "opa0.10" in ply.name:
                        clean_ply = str(ply)
                        break
                if not clean_ply:
                    clean_ply = str(ply_files[0])

        if not clean_ply:
            print("[Step6] No clean PLY found, skipping export.")
            return

        splat_path = self.work_dir / f"{scene_name}.splat"
        if not splat_path.exists() or not cfg.skip_existing:
            from .splat_export import export_from_ply
            export_from_ply(
                clean_ply, str(splat_path),
                sort_by_depth=True,
                max_gaussians=cfg.max_web_gaussians,
            )
        self._step_results["splat"] = str(splat_path)

        if cfg.render_viewer:
            viewer_path = self.work_dir / f"{scene_name}_viewer.html"
            if not viewer_path.exists() or not cfg.skip_existing:
                from .web_viewer import generate_viewer
                title = cfg.viewer_title or scene_name.replace("_", " ")
                generate_viewer(
                    str(splat_path), str(viewer_path),
                    title=title,
                    embed_splat=cfg.viewer_embed_splat,
                )
            self._step_results["viewer"] = str(viewer_path)

        if cfg.render_showcase:
            showcase_path = self.work_dir / f"{scene_name}_showcase.mp4"
            if not showcase_path.exists() or not cfg.skip_existing:
                ckpt = self._step_results.get("checkpoint")
                if not ckpt:
                    ckpts = sorted(self.train_dir.glob("checkpoint_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
                    ckpt = str(ckpts[-1]) if ckpts else None

                if ckpt and Path(ckpt).exists():
                    from .export import render_trajectory_video
                    try:
                        render_trajectory_video(
                            ckpt, str(showcase_path),
                            trajectory=cfg.showcase_trajectory,
                            num_frames=int(cfg.showcase_duration * 30),
                            resolution=(1920, 1080),
                            fps=30,
                        )
                        self._step_results["showcase"] = str(showcase_path)
                    except Exception as e:
                        print(f"[Step6] Showcase render failed: {e}")

        print(f"[Step6] Export complete.")

    def _print_deliverables(self):
        """List all output files for delivery."""
        print("\n  Deliverables:")
        deliverables = []

        # PLY files from cleanup
        for ply in sorted(self.clean_dir.glob("*.ply")):
            size_mb = ply.stat().st_size / (1024 * 1024)
            label = "PLY-STD" if "_standard" in ply.name else "PLY-WEB"
            deliverables.append(f"    [{label}] {ply.relative_to(self.work_dir)} ({size_mb:.1f} MB)")

        # Training checkpoints
        for ckpt in sorted(self.train_dir.glob("checkpoint_step*.pt"), key=lambda p: int(p.stem.split("step")[-1])):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            deliverables.append(f"    [CKPT] {ckpt.relative_to(self.work_dir)} ({size_mb:.1f} MB)")

        # Metadata
        meta = self.aligned_dir / "metadata.json"
        if meta.exists():
            deliverables.append(f"    [META] {meta.relative_to(self.work_dir)}")

        sr_meta = self.sr_dir / "sr_manifest.json"
        if sr_meta.exists():
            deliverables.append(f"    [SR] {sr_meta.relative_to(self.work_dir)}")

        for key, label in [("viewer", "VIEWER"), ("splat", "SPLAT"), ("showcase", "VIDEO")]:
            path = self._step_results.get(key)
            if path and Path(str(path)).exists():
                size_mb = Path(str(path)).stat().st_size / (1024 * 1024)
                deliverables.append(f"    [{label}] {Path(str(path)).relative_to(self.work_dir)} ({size_mb:.1f} MB)")

        if not deliverables:
            print("    No deliverables found.")
        else:
            for d in deliverables:
                print(d)

    def export_config(self, path: Optional[str] = None):
        """Export the current pipeline config as JSON for reproducibility."""
        if path is None:
            path = self.work_dir / "pipeline_config.json"
        cfg_dict = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in self.config.__dict__.items()
        }
        with open(path, "w") as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        print(f"Config exported to {path}")
