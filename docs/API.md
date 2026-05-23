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
    train_max_steps=25000,
)

pipeline = Pipeline(cfg)
pipeline.run()
```

## Stable Building Blocks

Many workflows are currently easier to compose through scripts:

- `scripts/assess_scene_inputs.py`
- `scripts/run_video_pipeline.py`
- `scripts/cluster_clean_ply.py`
- `scripts/filter_ply_confidence.py`
- `scripts/publish_clean_candidate.py`
- `scripts/validate_output.py`

Prefer these for reproducible GitHub issues and pull requests.
