# Python API

The command-line scripts are the most stable interface today. The Python API is
kept lightweight and may change while the project is in alpha.

## Video Pipeline

```python
from sr_3dgs import VideoPipeline, VideoPipelineConfig

cfg = VideoPipelineConfig(
    video_path="input_videos/object.mp4",
    work_dir="workspace_video",
    output_name="object",
    preset="standard",
    object_mask="auto",
    cluster_clean=True,
)

pipeline = VideoPipeline(cfg)
result = pipeline.run()
print(result)
```

## Image Pipeline

```python
from sr_3dgs import Pipeline, PipelineConfig

cfg = PipelineConfig(
    input_dir="input_images/object",
    work_dir="workspace/object",
    sr_mode="model",
    sr_model="real-esrgan",
    sr_scale=2,
    train_max_steps=25000,
)

pipeline = Pipeline(cfg)
pipeline.run()
```

Set `sr_mode="off"` and `sr_scale=1` for a geometry-first run without learned
super-resolution. Set `sr_mode="resize"` for deterministic Lanczos upscaling
when comparing SR strategies without adding model hallucination risk. Learned
SR has timeout fallback controls via `sr_model_load_timeout_s` and
`sr_frame_timeout_s`; inspect `sr_images/sr_manifest.json` to distinguish the
requested scale from the effective scale used for training. Set
`sr_strict_model=True` when a learned-SR failure should raise instead of
copying original-resolution frames.

Video extraction can use adaptive coverage recovery through
`extract_adaptive=True`, `extract_min_frames=<count>`, and
`extract_min_span=<ratio>`. The selected threshold pass and selected-frame
timeline span are recorded in `frames/extraction_manifest.json`.

## Stable Building Blocks

Many workflows are currently easier to compose through scripts:

- `scripts/assess_scene_inputs.py`
- `scripts/run_video_pipeline.py`
- `scripts/cluster_clean_ply.py`
- `scripts/filter_ply_confidence.py`
- `scripts/publish_clean_candidate.py`
- `scripts/validate_output.py`

Prefer these for reproducible GitHub issues and pull requests.
