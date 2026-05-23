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

Only run SOG conversion when you intentionally want the web/mobile package:

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
