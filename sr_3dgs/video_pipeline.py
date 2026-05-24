"""Video-to-3DGS pipeline — the complete client delivery workflow.

Input:  A video file (MP4, MOV, etc., 10-60 seconds)
Output:
  1. 3DGS .splat file (compressed, for web)
  2. Self-contained HTML viewer (mobile-ready WebGL)
  3. Showcase trajectory video (MP4, for portfolio/social media)
  4. Clean PLY file (for client use in other tools)

This is the "black box" that turns a client video into a deliverable
3D model + viewer in one command.
"""

import os
import shutil
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .video_extractor import VideoFrameExtractor
from .pipeline import Pipeline, PipelineConfig
from .splat_export import SplatExporter
from .web_viewer import generate_viewer
from .utils import ensure_dir, check_dependencies, print_dep_check, get_gpu_memory
from .input_analyzer import analyze_video
from .quality import diagnose_paths, write_delivery_report
from .sog_export import export_sog, export_sog_viewer
from .sr_strategy import (
    adjust_strategy_for_model_preflight,
    recommend_sr_strategy,
    write_strategy,
)
from .step2_super_resolution import SuperResolutionProcessor


@dataclass
class VideoPipelineConfig:
    """Configuration for the video-to-3DGS pipeline."""

    # Input
    video_path: str = ""

    # Output
    work_dir: str = "workspace_video"
    final_output_dir: str = "output"
    output_name: str = ""                    # Name for output files (default: video filename)

    # Frame extraction
    projection: str = "auto"                   # auto | perspective | equirectangular
    extract_fps: float = 3.0                  # Frames per second to extract
    extract_min_sharpness: float = 80.0        # Laplacian variance threshold
    extract_min_frame_diff: float = 0.015      # Minimum difference between frames
    extract_max_frames: int = 250              # Maximum frames to extract
    extract_min_frames: int = 48               # Coverage target before relaxing filters
    extract_min_span: float = 0.80             # Timeline coverage target before accepting a pass
    extract_adaptive: bool = True              # Relax filters when coverage is low
    extract_target_long_edge: int = 1920       # Resize while extracting (0 = no resize)
    equirect_face_size: int = 1024
    equirect_faces: tuple = ("front", "right", "back", "left")

    # Video time range (None = entire video)
    start_time: Optional[float] = None
    duration: Optional[float] = None

    # SR settings (passed through to PipelineConfig)
    sr_mode: str = "auto"                     # auto | off | resize | model
    sr_model: str = "real-esrgan"
    sr_scale: int = 4
    sr_device: str = "cuda"
    sr_kwargs: Dict[str, Any] = field(default_factory=dict)
    sr_model_load_timeout_s: int = 180
    sr_frame_timeout_s: int = 300
    sr_strict_model: bool = False
    sr_allow_download: bool = False

    # COLMAP settings
    colmap_camera_model: str = "SIMPLE_PINHOLE"
    colmap_gpu: int = 0

    # Training
    train_max_steps: int = 15_000
    train_warmup_steps: int = 500
    train_data_factor: int = 1
    train_max_render_dim: int = 1600
    train_device: str = "cuda"
    train_strategy: str = "default"
    train_cap_max: int = 200_000  # MCMCStrategy cap when train_strategy="mcmc"

    # Object-focused reconstruction
    object_bbox: str = ""                      # Optional left,top,right,bottom crop after alignment
    object_crop_pad_ratio: float = 0.18
    object_mask: str = "off"                   # off | auto
    object_mask_rect: str = ""                 # Optional x0,y0,x1,y1 in aligned/cropped image pixels
    object_mask_method: str = "fast"           # fast | grabcut | rembg
    mask_background_weight: float = 0.05
    mask_alpha_reg: float = 0.05

    # Cleanup
    cleanup_opacity_thresholds: tuple = (0.10, 0.50)
    web_opacity_threshold: float = 0.10
    cluster_clean: bool = False
    cluster_auto: bool = True
    cluster_preset: str = "balanced"
    cluster_opacity_min: float = 0.08
    cluster_voxel_size: float = 0.10
    cluster_max_radius: float = 0.0
    cluster_max_scale: float = 0.0
    cluster_select: str = "nearest"
    cluster_seed: str = "0,0,0"
    cluster_min_component_points: int = 1000
    cluster_dilate: int = 1

    # Web viewer
    viewer_title: str = "3D Gaussian Splatting"
    viewer_embed_splat: bool = True
    viewer_max_embed_mb: int = 40
    export_sog: bool = True
    generate_sog_viewer: bool = True
    sog_iterations: Optional[int] = None

    # Showcase video
    render_showcase: bool = True
    showcase_trajectory: str = "spiral"
    showcase_fps: int = 30
    showcase_resolution: tuple = (1920, 1080)
    showcase_duration: float = 8.0            # Seconds

    # Splat export
    max_web_gaussians: int = 2_000_000         # Cap for mobile performance
    build_delivery: bool = True
    publish_output: bool = True
    assess_inputs: bool = True


class VideoPipeline:
    """End-to-end video → 3D viewer pipeline.

    Usage:
        cfg = VideoPipelineConfig(video_path="client_video.mp4")
        pipeline = VideoPipeline(cfg)
        result = pipeline.run()

        # result["viewer_html"] -> path to HTML viewer
        # result["showcase_video"] -> path to showcase MP4
        # result["splat_file"] -> path to .splat file
        # result["clean_ply"] -> path to clean PLY
    """

    def __init__(self, config: VideoPipelineConfig):
        self.config = config
        self._setup()

    def _setup(self):
        cfg = self.config
        video_name = cfg.output_name or Path(cfg.video_path).stem
        # Sanitize
        video_name = "".join(c if c.isalnum() or c in "_-" else "_"
                            for c in video_name)
        self.video_name = video_name
        self.work_dir = Path(cfg.work_dir) / video_name
        ensure_dir(self.work_dir)

        # Sub-directories
        self.frames_dir = self.work_dir / "frames"
        self.manifest_path = self.work_dir / "input_manifest.json"
        self.splat_output = self.work_dir / f"{video_name}.splat"
        self.viewer_output = self.work_dir / f"{video_name}_viewer.html"
        self.showcase_output = self.work_dir / f"{video_name}_showcase.mp4"
        self.clean_ply = self.work_dir / "clean_output" / f"clean_opa{self.config.web_opacity_threshold:.2f}.ply"
        self.standard_ply = self.work_dir / "clean_output" / f"clean_opa{self.config.web_opacity_threshold:.2f}_standard.ply"
        self.sog_output = self.work_dir / f"{video_name}.sog"
        self.sog_viewer_output = self.work_dir / f"{video_name}_sog.html"
        self.delivery_dir = self.work_dir / "delivery"
        self.final_output_dir = Path(cfg.final_output_dir) / video_name
        self.reports_dir = self.work_dir / "reports"
        self.sr_dir = self.work_dir / "sr_images"
        self.sr_strategy_path = self.reports_dir / "sr_strategy.json"

    def run(self, skip_frame_extraction: bool = False,
            skip_dep_check: bool = False) -> Dict[str, str]:
        """Run the full video-to-3DGS pipeline.

        Returns dict with paths to all deliverables.
        """
        cfg = self.config

        if not skip_dep_check:
            status = check_dependencies()
            # For video pipeline, ffmpeg is required
            if status.get("ffmpeg") != "OK":
                status["ok"] = False
                status["missing"].append("ffmpeg (apt install ffmpeg)")
            print_dep_check(status)
            if not status["ok"]:
                raise RuntimeError(
                    "Missing required dependencies.\n"
                    "  Local: bash scripts/setup_local.sh\n"
                    "  AutoDL: bash scripts/setup_autodl.sh"
                )
        t0 = time.time()

        print("=" * 60)
        print(f"  Video → 3DGS Pipeline")
        print(f"  Input: {cfg.video_path}")
        print(f"  Output: {self.work_dir}")
        print("=" * 60)

        results = {}
        input_info = analyze_video(cfg.video_path, projection=cfg.projection)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(input_info.to_dict(), f, indent=2)
        results["input_manifest"] = str(self.manifest_path)
        print(f"[Input] {input_info.width}x{input_info.height}, "
              f"{input_info.duration:.1f}s, {input_info.projection} "
              f"({input_info.reason})")

        # ── Phase 1: Extract frames ──
        print(f"\n{'─'*40}")
        print(f"  Phase 1: Frame Extraction")
        print(f"{'─'*40}")

        if not skip_frame_extraction:
            extractor = VideoFrameExtractor(cfg.video_path, str(self.frames_dir))
            if input_info.projection == "equirectangular":
                per_face_budget = max(1, cfg.extract_max_frames // max(1, len(cfg.equirect_faces)))
                frames = extractor.extract_equirectangular_cubefaces(
                    fps=cfg.extract_fps,
                    min_sharpness=cfg.extract_min_sharpness,
                    min_frame_diff=cfg.extract_min_frame_diff,
                    max_source_frames=per_face_budget,
                    min_source_frames=max(
                        2,
                        cfg.extract_min_frames // max(1, len(cfg.equirect_faces)),
                    ),
                    min_span=cfg.extract_min_span,
                    adaptive=cfg.extract_adaptive,
                    face_size=cfg.equirect_face_size,
                    faces=tuple(cfg.equirect_faces),
                    start_time=cfg.start_time,
                    duration=cfg.duration,
                )
            else:
                frames = extractor.extract(
                    fps=cfg.extract_fps,
                    min_sharpness=cfg.extract_min_sharpness,
                    min_frame_diff=cfg.extract_min_frame_diff,
                    max_frames=cfg.extract_max_frames,
                    min_frames=cfg.extract_min_frames,
                    min_span=cfg.extract_min_span,
                    adaptive=cfg.extract_adaptive,
                    target_long_edge=cfg.extract_target_long_edge,
                    start_time=cfg.start_time,
                    duration=cfg.duration,
                )
            print(f"[Phase 1] Extracted {len(frames)} frames")
        else:
            frames = sorted(
                p for p in self.frames_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            print(f"[Phase 1] Using {len(frames)} existing frames")

        results["frames"] = str(self.frames_dir)
        results["frame_count"] = len(frames)
        extraction_manifest = self.frames_dir / "extraction_manifest.json"
        if extraction_manifest.exists():
            results["extraction_manifest"] = str(extraction_manifest)
        frame_quality = self._assess_inputs(
            results,
            image_dir=self.frames_dir,
            report_name="input_quality_frames",
        )
        if frame_quality is not None and extraction_manifest.exists():
            try:
                frame_quality["extraction"] = json.loads(
                    extraction_manifest.read_text(encoding="utf-8")
                )
            except Exception as exc:
                print(f"[SRStrategy] WARNING: failed to read extraction manifest: {exc}")
        effective_sr_mode = cfg.sr_mode
        effective_sr_model = cfg.sr_model
        effective_sr_scale = cfg.sr_scale
        if cfg.sr_mode == "auto":
            strategy = recommend_sr_strategy(
                frame_quality,
                preferred_model=cfg.sr_model,
                preferred_scale=cfg.sr_scale,
                vram_gb=get_gpu_memory().get("total_gb", 0.0),
            )
            if strategy.mode == "model":
                preflight = SuperResolutionProcessor(
                    image_dir=str(self.frames_dir),
                    output_dir=str(self.sr_dir),
                    sr_model_name=strategy.model,
                    scale=strategy.scale,
                    device=cfg.sr_device,
                    model_kwargs=cfg.sr_kwargs,
                    mode="model",
                )._model_preflight()
                strategy = adjust_strategy_for_model_preflight(
                    strategy,
                    preflight,
                    allow_download=cfg.sr_allow_download,
                )
            effective_sr_mode = strategy.mode
            effective_sr_model = strategy.model
            effective_sr_scale = strategy.scale
            write_strategy(self.sr_strategy_path, strategy)
            results["sr_strategy"] = str(self.sr_strategy_path)
            print(
                "[SRStrategy] "
                f"{strategy.mode} / {strategy.model} x{strategy.scale}: "
                f"{strategy.reason}"
            )

        # ── Phase 2: SR + 3DGS (reuse existing pipeline) ──
        print(f"\n{'─'*40}")
        print(f"  Phase 2: SR + 3DGS Reconstruction")
        print(f"{'─'*40}")

        pipeline_cfg = PipelineConfig(
            input_dir=str(self.frames_dir),
            work_dir=str(self.work_dir),
            sr_mode=effective_sr_mode,
            sr_model=effective_sr_model,
            sr_scale=effective_sr_scale,
            sr_device=cfg.sr_device,
            sr_kwargs=cfg.sr_kwargs,
            sr_model_load_timeout_s=cfg.sr_model_load_timeout_s,
            sr_frame_timeout_s=cfg.sr_frame_timeout_s,
            sr_strict_model=cfg.sr_strict_model,
            colmap_camera_model=cfg.colmap_camera_model,
            colmap_gpu=cfg.colmap_gpu,
            train_max_steps=cfg.train_max_steps,
            train_warmup_steps=cfg.train_warmup_steps,
            train_data_factor=cfg.train_data_factor,
            train_strategy=cfg.train_strategy,
            train_cap_max=cfg.train_cap_max,
            train_max_render_dim=cfg.train_max_render_dim,
            train_sh_degree=2,
            train_device=cfg.train_device,
            cleanup_opacity_thresholds=cfg.cleanup_opacity_thresholds,
            train_mask_background_weight=cfg.mask_background_weight,
            train_mask_alpha_reg=cfg.mask_alpha_reg,
        )

        pipeline = Pipeline(pipeline_cfg)
        pipeline.run(start_step=1, end_step=3)
        sr_manifest = pipeline._step_results.get("sr_manifest")
        if sr_manifest:
            results["sr_manifest"] = str(sr_manifest)
        if cfg.object_bbox:
            self._crop_aligned_for_object(pipeline)
        if cfg.object_mask == "auto":
            self._build_object_masks(pipeline)
            self._assess_inputs(
                results,
                image_dir=pipeline.aligned_dir / "images",
                mask_dir=self.work_dir / "object_masks",
                report_name="input_quality_object",
            )
        pipeline.run(start_step=4, end_step=4)
        results["checkpoint"] = pipeline._step_results.get("checkpoint", "")
        training_summary = pipeline._step_results.get("training_summary")
        if training_summary:
            results["training_summary"] = str(training_summary)
        pipeline.run_step5()
        self._cluster_clean_standard_ply(results)

        # ── Phase 3: Export .splat ──
        print(f"\n{'─'*40}")
        print(f"  Phase 3: Web Export")
        print(f"{'─'*40}")

        self._export_splat(results)
        if self.standard_ply.exists():
            results["standard_ply"] = str(self.standard_ply)

        # ── Phase 4: Generate viewer ──
        print(f"\n{'─'*40}")
        print(f"  Phase 4: Web Viewer")
        print(f"{'─'*40}")

        self._generate_viewer(results)
        self._export_sog(results)
        self._write_quality_report(results)
        self._publish_final_output(results)

        # ── Phase 5: Render showcase video ──
        if cfg.render_showcase:
            print(f"\n{'─'*40}")
            print(f"  Phase 5: Showcase Video")
            print(f"{'─'*40}")

            self._render_showcase(results)

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"  Pipeline complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")
        print(f"\n  Client Deliverables:")
        for key, path in results.items():
            if Path(str(path)).exists() and key not in ("frames", "frame_count"):
                size = Path(str(path)).stat().st_size / (1024 * 1024)
                print(f"    [{key}] {path} ({size:.1f} MB)")
        print(f"{'=' * 60}")

        return results

    def _write_quality_report(self, results: Dict):
        cfg = self.config
        targets = []
        for key in ("splat_file", "standard_ply"):
            path = results.get(key)
            if path and Path(path).exists():
                targets.append(path)
        if not targets:
            return

        report_dir = self.work_dir / "reports"
        ensure_dir(report_dir)
        report_path = report_dir / "diagnostics.json"
        report = diagnose_paths(targets)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        results["diagnostics"] = str(report_path)

        if cfg.build_delivery:
            write_delivery_report(
                delivery_dir=self.delivery_dir,
                scene_name=self.video_name,
                results=results,
                diagnostics=report,
            )
            results["delivery"] = str(self.delivery_dir)

    def _assess_inputs(self, results: Dict, image_dir, report_name: str, mask_dir=None):
        if not self.config.assess_inputs:
            return
        try:
            from scripts.assess_scene_inputs import assess_scene, write_html
        except Exception as exc:
            print(f"[InputQuality] WARNING: quality assessor unavailable: {exc}")
            return

        report_dir = self.reports_dir
        ensure_dir(report_dir)
        json_path = report_dir / f"{report_name}.json"
        html_path = report_dir / f"{report_name}.html"
        try:
            report = assess_scene(
                self.work_dir,
                images_dir=image_dir,
                masks_dir=mask_dir,
            )
            json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            write_html(report, html_path)
            results[report_name] = str(json_path)
            verdict = report.get("verdict", {})
            print(
                "[InputQuality] "
                f"{report_name}: {verdict.get('score', 'n/a')}/100 "
                f"({', '.join(verdict.get('problems') or ['no major input problems'])})"
            )
            return report
        except Exception as exc:
            print(f"[InputQuality] WARNING: failed to assess {report_name}: {exc}")
            return None

    def _export_splat(self, results: Dict):
        """Export checkpoint to .splat format."""
        cfg = self.config

        # Try checkpoint first, then PLY
        ckpt = results.get("checkpoint", "")
        if ckpt and Path(ckpt).exists():
            exporter = SplatExporter(ckpt)
            exporter.export(
                str(self.splat_output),
                sort_by_depth=True,
                opacity_threshold=0.01,
                max_gaussians=cfg.max_web_gaussians,
            )
        else:
            # Try PLY from training output
            ply_candidates = sorted(self.work_dir.glob("train_output/*.ply"))
            ckpt_candidates = sorted(self.work_dir.glob("train_output/checkpoint_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
            clean_candidates = sorted(self.work_dir.glob("clean_output/*.ply"))

            if clean_candidates:
                src_ply = clean_candidates[0]
            elif ply_candidates:
                src_ply = ply_candidates[-1]
            elif ckpt_candidates:
                exporter = SplatExporter(str(ckpt_candidates[-1]))
                exporter.export(
                    str(self.splat_output),
                    sort_by_depth=True,
                    opacity_threshold=0.01,
                    max_gaussians=cfg.max_web_gaussians,
                )
                results["splat_file"] = str(self.splat_output)
                return
            else:
                print("[Phase 3] WARNING: No checkpoint or PLY found.")
                return

            from .splat_export import export_from_ply
            export_from_ply(
                str(src_ply), str(self.splat_output),
                sort_by_depth=True,
                max_gaussians=cfg.max_web_gaussians,
            )

        results["splat_file"] = str(self.splat_output)

    def _crop_aligned_for_object(self, pipeline):
        from scripts.crop_aligned_object import crop_aligned

        bbox = tuple(int(v) for v in self.config.object_bbox.split(","))
        if len(bbox) != 4:
            raise ValueError("object_bbox must be left,top,right,bottom")

        cropped = self.work_dir / "aligned_object"
        crop_aligned(
            pipeline.aligned_dir,
            cropped,
            pad_ratio=self.config.object_crop_pad_ratio,
            bbox=bbox,
        )
        pipeline.aligned_dir = cropped
        pipeline._step_results["aligned"] = str(cropped)
        print(f"[ObjectCrop] Using object-aligned data: {cropped}")

    def _build_object_masks(self, pipeline):
        from scripts.build_object_masks import build_masks

        mask_dir = self.work_dir / "object_masks"
        rect = self.config.object_mask_rect or None
        if rect is None and self.config.object_bbox:
            # After crop, the object usually fills most of the image. Use a
            # generous in-crop prior instead of the original-frame bbox.
            rect = None
        meta = build_masks(
            pipeline.aligned_dir,
            mask_dir,
            rect=rect,
            force=True,
            method=self.config.object_mask_method,
        )
        pipeline.config.train_mask_dir = str(mask_dir)
        print(f"[ObjectMask] Built {meta['count']} masks at {mask_dir}")

    def _cluster_clean_standard_ply(self, results: Dict):
        cfg = self.config
        if not cfg.cluster_clean:
            return
        if not self.standard_ply.exists():
            print("[ClusterClean] WARNING: No standard PLY, skipping cluster cleanup.")
            return

        from scripts.cluster_clean_ply import _parse_vec3, clean_ply

        src = self.standard_ply
        dst = src.with_name(src.stem + "_cluster.ply")
        report = self.work_dir / "reports" / "cluster_clean.json"
        meta = clean_ply(
            src,
            dst,
            report_path=report,
            opacity_min=cfg.cluster_opacity_min,
            max_radius=cfg.cluster_max_radius,
            max_scale=cfg.cluster_max_scale,
            voxel_size=cfg.cluster_voxel_size,
            select=cfg.cluster_select,
            seed=_parse_vec3(cfg.cluster_seed),
            min_component_points=cfg.cluster_min_component_points,
            dilate=max(0, cfg.cluster_dilate),
            auto=cfg.cluster_auto,
            preset=cfg.cluster_preset,
        )
        self.standard_ply = dst
        results["cluster_clean_report"] = str(report)
        print(
            "[ClusterClean] Kept "
            f"{meta['output_count']}/{meta['input_count']} Gaussians "
            f"({meta['removed_percent']:.1f}% removed): {dst}"
        )

    def _generate_viewer(self, results: Dict):
        """Generate the self-contained HTML viewer."""
        cfg = self.config
        splat_path = results.get("splat_file", str(self.splat_output))

        if not Path(splat_path).exists():
            print("[Phase 4] WARNING: No .splat file, skipping viewer.")
            return

        generate_viewer(
            splat_path=splat_path,
            output_html=str(self.viewer_output),
            title=cfg.viewer_title or self.video_name,
            embed_splat=cfg.viewer_embed_splat,
            max_splat_mb=cfg.viewer_max_embed_mb,
        )

        results["viewer_html"] = str(self.viewer_output)

    def _export_sog(self, results: Dict):
        cfg = self.config
        if not cfg.export_sog:
            return
        if not self.standard_ply.exists():
            print("[SOG] WARNING: No standard PLY, skipping SOG export.")
            return

        export_sog(
            str(self.standard_ply),
            str(self.sog_output),
            overwrite=True,
            iterations=cfg.sog_iterations,
        )
        results["sog_file"] = str(self.sog_output)

        if cfg.generate_sog_viewer:
            export_sog_viewer(
                str(self.standard_ply),
                str(self.sog_viewer_output),
                overwrite=True,
                unbundled=True,
                iterations=cfg.sog_iterations,
            )
            results["sog_viewer_html"] = str(self.sog_viewer_output)

    def _publish_final_output(self, results: Dict):
        if not self.config.publish_output:
            return
        try:
            from scripts.publish_output import publish
            manifest = publish(self.work_dir, self.final_output_dir, self.video_name, self.video_name)
            results["final_output"] = str(self.final_output_dir)
            print(f"[Output] Published final deliverables: {self.final_output_dir}")
            print(f"[Output] Open first: {manifest.get('open_first', 'START_HERE.html')}")
        except Exception as exc:
            print(f"[Output] WARNING: final output publish failed: {exc}")

    def _render_showcase(self, results: Dict):
        """Render a showcase trajectory video."""
        ckpt = results.get("checkpoint", "")
        if not ckpt:
            ckpts = sorted(self.work_dir.glob("train_output/checkpoint_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
            ckpt = str(ckpts[-1]) if ckpts else None

        if not ckpt:
            print("[Phase 5] WARNING: No checkpoint, skipping showcase video.")
            return

        from .export import render_trajectory_video

        cfg = self.config
        render_trajectory_video(
            checkpoint_path=ckpt,
            output_path=str(self.showcase_output),
            trajectory=cfg.showcase_trajectory,
            num_frames=int(cfg.showcase_duration * cfg.showcase_fps),
            resolution=cfg.showcase_resolution,
            fps=cfg.showcase_fps,
        )

        results["showcase_video"] = str(self.showcase_output)
