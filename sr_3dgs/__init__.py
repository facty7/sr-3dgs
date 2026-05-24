from .pipeline import Pipeline, PipelineConfig
from .step1_colmap import COLMAPExtractor
from .step2_super_resolution import SuperResolutionProcessor
from .step3_intrinsic_align import IntrinsicAligner
from .step4_train_3dgs import SR3DGSTrainer
from .step5_cleanup import CleanupProcessor
from .video_pipeline import VideoPipeline, VideoPipelineConfig
from .video_extractor import VideoFrameExtractor
from .splat_export import SplatExporter, export_from_ply
from .web_viewer import generate_viewer
from .export import export_to_ply, render_trajectory_video
from .sr_strategy import (
    SRStrategy,
    adjust_strategy_for_model_preflight,
    recommend_sr_strategy,
)

__version__ = '1.2.0'
__all__ = [
    # Image pipeline
    'Pipeline', 'PipelineConfig',
    'COLMAPExtractor', 'SuperResolutionProcessor',
    'IntrinsicAligner', 'SR3DGSTrainer', 'CleanupProcessor',
    # Video pipeline
    'VideoPipeline', 'VideoPipelineConfig',
    'VideoFrameExtractor',
    # Export
    'SplatExporter', 'export_from_ply',
    'export_to_ply', 'render_trajectory_video',
    # Web viewer
    'generate_viewer',
    # SR strategy
    'SRStrategy', 'adjust_strategy_for_model_preflight', 'recommend_sr_strategy',
]
