# SR 3DGS Pipeline

Object-focused pipeline for producing 3D Gaussian Splatting deliverables from
phone video or image collections.

[中文](README.zh-CN.md) | [日本語](README.ja.md)

![3DGS example preview](docs/assets/toy-preview-cleaned.png)

SR 3DGS Pipeline integrates capture preparation, frame selection, COLMAP camera
reconstruction, gsplat training, Gaussian cleanup, and web delivery packaging.
The project targets static object reconstruction workflows such as products,
collectibles, tabletop objects, toys, and small scanned assets.

The repository is an alpha-stage engineering pipeline built around established
open-source components. It does not introduce a new 3DGS research method, and
output quality remains dependent on capture coverage, focus, lighting, object
texture, and scene stability.

## Features

- Phone video and image-folder inputs.
- Perspective and equirectangular frame extraction.
- COLMAP-based camera reconstruction.
- gsplat training entry points.
- Optional object crop, automatic masks, and mask-aware training.
- Automatic cleanup for detached clusters, sparse floaters, and haze-like
  low-confidence splats.
- Flat `output/<scene>/` delivery folders.
- PlayCanvas SOG export for web and mobile preview.
- Standard PLY export for SuperSplat and downstream tooling.
- Input-quality reports for extracted frames and masks.
- CI-safe checks, output validation, delivery scoring, and local HTTP preview
  smoke tests.

## Example

The repository includes lightweight preview images for the current object demo.
Large generated assets such as PLY, SOG, videos, and reconstruction workspaces
are intentionally excluded from git.

![Point cloud contact sheet](docs/assets/toy-contact-sheet.png)

Published scene folders use this layout:

```text
output/<scene>/
  START_HERE.html
  preview.html
  <scene>_v<timestamp>.sog
  <scene>_high_quality.ply
  diagnostics.json
  manifest.json
```

Timestamped SOG names reduce stale browser-cache issues during repeated
publishing.

## Capture Requirements

Recommended input conditions:

- one primary static subject
- 20-60 seconds of slow orbiting phone video, or 40-180 usable still images
- broad angular coverage around the subject
- optional low, middle, and high camera-height bands
- stable exposure and lighting
- background visually distinct from the subject
- limited motion blur, reflections, transparency, and deformation

Living or deformable subjects are outside the main target unless they remain
effectively static during capture.

See [docs/CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md) for capture guidance.

## Installation

Base install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Training dependencies:

```bash
pip install -e ".[training]"
```

Optional mask dependencies:

```bash
pip install -r requirements-optional.txt
```

Backend check:

```bash
python scripts/check_backend.py
```

## Run From Video

```bash
python scripts/run_video_pipeline.py \
  --video input_videos/object.mp4 \
  --output_name object \
  --projection perspective \
  --preset standard \
  --object_mask auto \
  --cluster_clean
```

Super-resolution is optional. The default `standard` preset is geometry-first
and keeps extracted frames at their original resolution. Enable learned SR only
when the input is sharp enough and the machine has enough VRAM:

```bash
# No learned SR; fastest and least likely to hallucinate detail.
python scripts/run_video_pipeline.py --video input_videos/object.mp4 \
  --preset standard --sr_mode off --sr_scale 1

# Deterministic upscale for ablation tests.
python scripts/run_video_pipeline.py --video input_videos/object.mp4 \
  --preset standard --sr_mode resize --sr_scale 2

# Learned SR before training.
python scripts/run_video_pipeline.py --video input_videos/object.mp4 \
  --preset quality --sr_mode model --sr_model real-esrgan --sr_scale 2
```

Each run writes `workspace_video/<scene>/sr_images/sr_manifest.json` so the
chosen SR mode, scale, output resolution, and fallback status are recorded.
Learned SR runs are guarded by load/progress timeouts; if the model cannot
load or make progress, the pipeline falls back to original-resolution frames
and records `effective_mode`, `effective_scale`, and `model_preflight` in the
manifest. Set `--sr_strict_model` to treat learned-SR failures as errors
instead of fallback events. In `--sr_mode auto`, learned SR is selected only
when the needed weights are already local; set `--sr_allow_download` to permit
first-run weight downloads.

Optional object crop:

```bash
--object_bbox left,top,right,bottom
```

Equirectangular video:

```bash
python scripts/run_video_pipeline.py \
  --video input_videos/object_360.mp4 \
  --output_name object_360 \
  --projection equirectangular \
  --equirect_face_size 1024 \
  --equirect_faces front,right,back,left \
  --object_mask auto \
  --cluster_clean
```

## Input Assessment

```bash
python scripts/assess_scene_inputs.py workspace_video/object \
  --report workspace_video/object/reports/input_quality.json \
  --html workspace_video/object/reports/input_quality.html
```

`run_video_pipeline.py` also writes:

- `workspace_video/<scene>/reports/input_quality_frames.html`
- `workspace_video/<scene>/reports/input_quality_object.html` when
  `--object_mask auto` is enabled

The assessment covers frame count, blur, near-duplicates, large viewpoint
jumps, foreground-mask size, and masks touching image boundaries.

## Validate And Preview

```bash
python scripts/validate_output.py output/object
python scripts/score_output.py output/object
python scripts/http_preview_smoke.py output/object
python scripts/serve_output.py --scene object --port 8765
```

Open the local preview:

```text
http://127.0.0.1:8765/output/object/START_HERE.html
```

Local HTTP serving is required for browser-based SOG loading; direct `file://`
preview is not reliable.

## Cleanup Tools

- `scripts/cluster_clean_ply.py`: primary-component filtering.
- `scripts/filter_ply_confidence.py`: low-confidence splat filtering.
- `scripts/crop_ply_by_core.py`: core-bounded candidate generation.
- `scripts/compare_clean_candidates.py`: candidate metric comparison.
- `scripts/render_ply_contact_sheet.py`: CPU contact-sheet rendering for visual
  review.

Cleanup modules reduce common reconstruction artifacts. They are not a
substitute for sufficient camera coverage, sharp input frames, stable lighting,
and static scene geometry.

## Repository Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ci_check.py
PYTHONDONTWRITEBYTECODE=1 python scripts/release_readiness.py
PYTHONDONTWRITEBYTECODE=1 python scripts/release_readiness.py --include_output
```

Generated reconstruction assets are excluded from version control. Source code,
configuration, documentation, tests, and lightweight preview images are tracked.

## Status

Implemented:

- video and image-folder orchestration
- perspective and equirectangular frame extraction
- COLMAP/pycolmap camera preparation
- gsplat training entry points
- standard PLY export
- PlayCanvas SOG viewer export
- object crop and mask-aware training hooks
- automatic Gaussian cleanup utilities
- input-quality reports
- output validation and local HTTP preview checks

Planned:

- broader public benchmark scenes
- improved segmentation backends
- stronger default-parameter coverage across capture devices
- optional visual QA workflows for dedicated machines

## Related Projects

- [COLMAP](https://colmap.github.io/)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [Nerfstudio / Splatfacto](https://docs.nerf.studio/)
- [PlayCanvas SuperSplat and SOG tooling](https://github.com/playcanvas)

See [docs/PROJECT_POSITIONING.md](docs/PROJECT_POSITIONING.md) for project
positioning.

## Contributing

Contributions are welcome for capture tests, cleanup methods, segmentation
backends, benchmark scenes, and documentation.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
