# Contributing

Contributions should improve the object-to-delivery workflow without requiring
large generated files in git.

## Before Opening A PR

Run the CI-safe checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ci_check.py
```

If you have local demo outputs:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/release_readiness.py --include_output
```

## Generated Files

Do not commit:

- `workspace_video/`
- `output/`
- videos
- PLY/SOG/SPLAT assets
- checkpoints
- `__pycache__`

Demo outputs should be regenerated locally or attached as release artifacts.

## Adding A New Scene

1. Put raw media under `input_videos/` or another ignored local folder.
2. Run the pipeline and publish final files to `output/<scene>`.
3. Add the scene to `configs/benchmark_outputs.json`.
4. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/benchmark_outputs.py \
  --config configs/benchmark_outputs.json
```

5. Include screenshots or notes in the PR, but do not commit large generated
   assets.

## Cleanup Changes

Changes to automatic cleanup should keep the synthetic test passing:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/test_cluster_clean.py
```

If a change improves one scene but hurts another, add both scenes to the
benchmark config so the tradeoff is visible.

## Heavy Checks

Chrome/Edge render screenshot QA can stress local machines. It is opt-in only.
Do not add it to default CI.
