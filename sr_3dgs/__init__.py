"""Public package exports for sr-3dgs.

The package keeps top-level imports lightweight so diagnostics, SR planning,
and file-conversion helpers do not require the training stack until it is used.
"""

from importlib import import_module

__version__ = "1.2.0"

_LAZY_EXPORTS = {
    "Pipeline": ("pipeline", "Pipeline"),
    "PipelineConfig": ("pipeline", "PipelineConfig"),
    "COLMAPExtractor": ("step1_colmap", "COLMAPExtractor"),
    "SuperResolutionProcessor": ("step2_super_resolution", "SuperResolutionProcessor"),
    "IntrinsicAligner": ("step3_intrinsic_align", "IntrinsicAligner"),
    "SR3DGSTrainer": ("step4_train_3dgs", "SR3DGSTrainer"),
    "CleanupProcessor": ("step5_cleanup", "CleanupProcessor"),
    "VideoPipeline": ("video_pipeline", "VideoPipeline"),
    "VideoPipelineConfig": ("video_pipeline", "VideoPipelineConfig"),
    "VideoFrameExtractor": ("video_extractor", "VideoFrameExtractor"),
    "SplatExporter": ("splat_export", "SplatExporter"),
    "export_from_ply": ("splat_export", "export_from_ply"),
    "generate_viewer": ("web_viewer", "generate_viewer"),
    "export_to_ply": ("export", "export_to_ply"),
    "render_trajectory_video": ("export", "render_trajectory_video"),
    "SRStrategy": ("sr_strategy", "SRStrategy"),
    "adjust_strategy_for_model_preflight": (
        "sr_strategy",
        "adjust_strategy_for_model_preflight",
    ),
    "recommend_sr_strategy": ("sr_strategy", "recommend_sr_strategy"),
}

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "COLMAPExtractor",
    "SuperResolutionProcessor",
    "IntrinsicAligner",
    "SR3DGSTrainer",
    "CleanupProcessor",
    "VideoPipeline",
    "VideoPipelineConfig",
    "VideoFrameExtractor",
    "SplatExporter",
    "export_from_ply",
    "export_to_ply",
    "render_trajectory_video",
    "generate_viewer",
    "SRStrategy",
    "adjust_strategy_for_model_preflight",
    "recommend_sr_strategy",
]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
