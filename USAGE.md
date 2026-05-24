# Usage Notes

This file collects command patterns that are useful after reading the main
README. It avoids project-specific local paths and does not assume any private
workspace layout.

## Typical Object Workflow

```bash
python scripts/run_video_pipeline.py \
  --video input_videos/object.mp4 \
  --output_name object \
  --projection perspective \
  --preset standard \
  --object_mask auto \
  --cluster_clean
```

Inspect input reports:

```bash
python scripts/assess_scene_inputs.py workspace_video/object \
  --report workspace_video/object/reports/input_quality.json \
  --html workspace_video/object/reports/input_quality.html
```

Validate final output:

```bash
python scripts/validate_output.py output/object
python scripts/score_output.py output/object
python scripts/http_preview_smoke.py output/object
```

Preview locally:

```bash
python scripts/serve_output.py --scene object --port 8765
```

Open:

```text
http://127.0.0.1:8765/output/object/START_HERE.html
```

## Adaptive Frame Extraction

Video frame extraction is adaptive by default. It starts with the preset
blur, exposure, contrast, timeline-span, and diversity thresholds. It relaxes
sharpness/diversity only when too few frames survive for coverage. This is
meant for ordinary phone captures where motion blur, auto-exposure changes,
and near-duplicate frames vary across the clip.

```bash
python scripts/run_video_pipeline.py --video input_videos/object.mp4 \
  --preset standard --extract_min_frames 64 --extract_min_span 0.85
```

Use `--no_adaptive_extract` for strict ablation runs. Each run records the
selected extraction pass in
`workspace_video/<scene>/frames/extraction_manifest.json`. `extract_min_span`
is the required selected-frame coverage across the source timeline. This helps
ordinary turntable-style phone videos avoid runs where enough frames survived
but they came from only part of the orbit.
Very dark, overexposed, clipped, or near-flat frames are filtered before blur
and duplicate checks; tune this with `--extract_min_brightness`,
`--extract_max_brightness`, `--extract_min_contrast`, and
`--extract_max_clipped_ratio` for unusual footage.

## COLMAP Reconstruction Health

The pipeline writes `workspace_video/<scene>/colmap/colmap_report.json` after
camera reconstruction. It records the registered-image count, registration
ratio, sparse point count, selected camera model, and any fallback attempts.
If the primary model is weak, tune the fallback list:

```bash
python scripts/run_video_pipeline.py --video input_videos/object.mp4 \
  --preset standard \
  --colmap_camera SIMPLE_PINHOLE \
  --colmap_camera_fallbacks SIMPLE_RADIAL,PINHOLE
```

## Super-Resolution Modes

Use explicit SR modes when comparing output quality:

```bash
# Original extracted frames.
python scripts/run_video_pipeline.py --video input_videos/object.mp4 \
  --preset standard --sr_mode off --sr_scale 1

# Deterministic non-learned upscale.
python scripts/run_video_pipeline.py --video input_videos/object.mp4 \
  --preset standard --sr_mode resize --sr_scale 2

# Learned SR for sharp, small inputs.
python scripts/run_video_pipeline.py --video input_videos/object.mp4 \
  --preset quality --sr_mode model --sr_model real-esrgan --sr_scale 2
```

Check learned-SR model weights before a long run:

```bash
python scripts/check_sr_models.py --sr_model real-esrgan --sr_scale 2
```

Learned SR is guarded by load and progress timeouts. Tune them with
`--sr_model_load_timeout` and `--sr_frame_timeout`; `<=0` disables timeout
fallback and should be reserved for controlled debugging. Check
`workspace_video/<scene>/sr_images/sr_manifest.json`: `scale` is the requested
scale, while `effective_scale` is what actually reached training.
`model_preflight` records whether local weights were found or a download was
needed. Add `--sr_strict_model` when a learned-SR run should fail instead of
falling back. In `--sr_mode auto`, learned SR is selected only when the needed
weights are already local, extraction coverage is healthy, and the input report
does not show strong multi-view consistency risks such as near-duplicate
frames, exposure jumps, low contrast, or clipped frames. The auto decision
stores these risk flags in `reports/sr_strategy.json`; add `--sr_allow_download`
to permit automatic weight downloads.

Plan a reproducible sweep without starting heavy jobs:

```bash
python scripts/plan_sr_sweep.py \
  --video input_videos/object.mp4 \
  --preset standard \
  --cluster_clean \
  --no_showcase
```

The plan records each run command, its expected workspace/output folders, and
a ready-to-run summary command. Add `--run` only for launching all planned
runs.

For ordinary phone captures, include extraction coverage variants in the same
plan:

```bash
python scripts/plan_sr_sweep.py \
  --video input_videos/object.mp4 \
  --preset standard \
  --phone_coverage_sweep \
  --cluster_clean \
  --no_showcase
```

This crosses the SR strategies with `cover64`, `cover96`, and `strict64`
frame-extraction variants. Add custom variants with
`--extract_variant name:min_frames:max_frames[:fps][:span][:adaptive|strict]`.
Values up to `1.0` are treated as timeline span targets; prefix low frame rates
with `fps`, for example `name:80:240:fps0.5:span0.90:adaptive`.

After the runs finish, summarize the delivery metrics and SR metadata:

```bash
python scripts/summarize_sr_sweep.py \
  --plan workspace_video/sr_sweeps/sr_sweep_plan.json \
  --report workspace_video/sr_sweeps/sr_sweep_summary.json
```

The plan-based summary automatically skips runs whose output folder does not
exist yet. You can also summarize explicit output folders:

```bash
python scripts/summarize_sr_sweep.py \
  output/sr_sweeps/object_off_x1 \
  output/sr_sweeps/object_resize_x2 \
  output/sr_sweeps/object_auto_x2 \
  output/sr_sweeps/object_model_real-esrgan_x2 \
  --work_root workspace_video/sr_sweeps \
  --report workspace_video/sr_sweeps/sr_sweep_summary.json
```

The summary table reports `eff_x`, the actual SR scale inferred from the
manifest. A `model` run that falls back safely can therefore show requested
`scale=2` with `eff_x=1`. The `risk` column counts SR consistency flags from
`reports/sr_strategy.json`. The `frames`, `target`, `cov`, `span`, and `pass`
columns come from `frames/extraction_manifest.json`; `cam` and `cam_ratio`
come from `colmap/colmap_report.json`. Low-coverage or weak-COLMAP phone
captures do not outrank better-covered runs only because their delivery score
or PSNR is higher. `span` is the selected-frame coverage across the source
timeline; for turntable-style phone videos it should usually be close to
`1.00`. The `fb` column marks SR fallback runs.

The JSON report includes a lightweight `analysis` block with a recommended
output, review notes, and `action_items`. These action items separate common
failure causes such as weak frame coverage, partial timeline coverage, COLMAP
registration trouble, learned-SR runtime fallback, SR consistency risk, low
point count, oversized mobile assets, and incomplete delivery packaging. Treat
them as triage guidance before spending time on manual visual comparison.

## Candidate Cleanup

Generate and compare cleanup candidates before running heavy SOG conversion.

```bash
python scripts/filter_ply_confidence.py \
  output/object/object_high_quality.ply \
  workspace_video/object/candidates/object_filtered.ply \
  --report workspace_video/object/candidates/object_filtered.json

python scripts/compare_clean_candidates.py \
  --base output/object \
  workspace_video/object/candidates/object_filtered.ply \
  --report workspace_video/object/candidates/comparison.json
```

Render a quick contact sheet:

```bash
python scripts/render_ply_contact_sheet.py \
  workspace_video/object/candidates/object_filtered.ply \
  --out workspace_video/object/candidates/object_filtered_contact.png \
  --title object-filtered
```

## Publishing A Clean Candidate

First publish a lightweight review folder:

```bash
python scripts/publish_clean_candidate.py \
  workspace_video/object/candidates/object_filtered.ply \
  --out workspace_video/object/review/object_filtered \
  --scene_name object \
  --asset_name object
```

Run SOG conversion only when the web/mobile package is required:

```bash
python scripts/preflight_heavy.py \
  workspace_video/object/candidates/object_filtered.ply \
  --step sog

python scripts/publish_clean_candidate.py \
  workspace_video/object/candidates/object_filtered.ply \
  --out output/object \
  --scene_name object \
  --asset_name object \
  --replace \
  --convert_sog \
  --i_understand_this_is_heavy
```

## Repository Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ci_check.py
PYTHONDONTWRITEBYTECODE=1 python scripts/release_readiness.py
PYTHONDONTWRITEBYTECODE=1 python scripts/release_readiness.py --include_output
```

## Generated Files

Keep generated assets out of git:

- `workspace_video/`
- `output/`
- `input_videos/`
- `.ply`, `.sog`, `.splat`, videos, and checkpoints
