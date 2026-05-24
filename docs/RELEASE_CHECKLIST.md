# Release Checklist

This project has two release modes.

## Source Publish

Use this before pushing the repository to GitHub. It does not require large
generated outputs.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/release_readiness.py
```

Required guarantees:

- source files, configs, docs, and CI workflow are present
- Python syntax parses without writing bytecode
- repository hygiene audit passes
- synthetic auto-clean test passes
- no browser, training, SOG conversion, or GPU work is launched

## Local Demo Release

Use this on a workstation that has already generated `output/toy`.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/release_readiness.py --include_output
```

Additional guarantees:

- `output/toy` contains final deliverables only
- old demo versions are archived under `workspace_video/toy/archived_outputs`
- `output/START_HERE.html` points to current final deliveries
- `START_HERE.html`, `preview.html`, SOG, PLY, settings, and diagnostics exist
- SOG size and PLY point count pass the benchmark gates
- benchmark summary includes the toy scene

## Heavy Checks

Chrome/Edge render screenshot QA is opt-in only:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/qa_render_chrome.py \
  http://127.0.0.1:8765/output/toy/preview.html \
  --enable_heavy_browser
```

Run this only on an idle machine when WebGL screenshot evidence is required. It
is not part of default CI or smoke checks.

## Quality Bar

The current benchmark gates are defined in `configs/benchmark_outputs.json`:

- SOG <= 12 MB for mobile/web delivery
- PLY point count >= 120k
- score >= 85

These gates are smoke-level checks, not a substitute for visual review. Add
more scenes to the benchmark config as soon as more public or user-approved
test captures are available.

## Publishing Candidates

Cleaned PLY candidates can be promoted without running heavy conversion:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/publish_clean_candidate.py \
  workspace_video/toy/cleanup_candidates/scale006/toy_high_quality_scale006.ply \
  --out workspace_video/toy/review_candidates/scale006 \
  --scene_name toy \
  --asset_name toy
```

Compare candidates before conversion:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/compare_clean_candidates.py \
  --base output/toy \
  workspace_video/toy/cleanup_candidates/scale006/toy_high_quality_scale006.ply \
  workspace_video/toy/cleanup_candidates/scale005/toy_high_quality_scale005.ply \
  --report workspace_video/toy/review_candidates/candidate_comparison.json
```

For close-to-object haze or thin detached sheets, generate core-crop review
candidates before spending time on SOG conversion:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/crop_ply_by_core.py \
  output/toy/toy_high_quality.ply \
  workspace_video/toy/core_crop_candidates/p01_99_m08/toy_high_quality_core_p01_99_m08.ply \
  --report workspace_video/toy/core_crop_candidates/p01_99_m08/report.json \
  --core_opacity_percentile 90 \
  --lower_percentile 1 \
  --upper_percentile 99 \
  --margin 0.08
```

Use `--convert_sog` only when the web/mobile SOG package is required. Use
`--replace` to archive an existing final output before replacing it.

Heavy conversion requires both preflight and an explicit confirmation flag:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/preflight_heavy.py \
  workspace_video/toy/cleanup_candidates/scale006/toy_high_quality_scale006.ply \
  --step sog

PYTHONDONTWRITEBYTECODE=1 python scripts/publish_clean_candidate.py \
  workspace_video/toy/cleanup_candidates/scale006/toy_high_quality_scale006.ply \
  --out output/toy \
  --scene_name toy \
  --asset_name toy \
  --replace \
  --convert_sog \
  --i_understand_this_is_heavy
```
