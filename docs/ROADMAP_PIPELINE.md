# Pipeline Roadmap

## Decision

This project should be a pipeline orchestrator, not a fully custom 3DGS stack.

Keep custom code for:

- input inspection and frame extraction
- perspective/360 routing
- COLMAP/SfM data preparation
- object masks and object-focused training glue
- diagnostics and delivery packaging
- format conversion glue

Prefer established tools for:

- 3DGS rasterization and strategies: gsplat
- full training baselines: Nerfstudio Splatfacto, OpenSplat
- editing and publishing: SuperSplat, PlayCanvas
- lightweight web formats: SOG, SPZ

## Current Flow

```text
video/images
  -> input_manifest.json
  -> frame extraction
  -> COLMAP
  -> aligned scene data
  -> optional object crop
  -> optional object masks
  -> mask-aware gsplat training
  -> cleanup
  -> standard PLY
  -> SOG + official PlayCanvas viewer
  -> output/<scene>/
```

## Deliverables

Final deliverables should be flat and easy to find:

```text
output/<scene>/
  START_HERE.html
  preview.html
  <scene>_v<timestamp>.sog
  <scene>_high_quality.ply or toy_high_quality.ply
  diagnostics.json
  manifest.json
```

Intermediate artifacts stay in:

```text
workspace_video/<scene>/
```

## Quality Gates

Do not publish a scene unless diagnostics pass:

- no NaN or Inf coordinates
- coordinate radius is sane
- no extreme coordinate outliers
- standard PLY keeps log-scales and opacity logits
- SOG viewer opens successfully

## Next Milestones

1. Add a robust segmentation backend:
   - fast heuristic masks as default
   - SAM/ONNX segmentation as optional quality mode
2. Add object-cluster cleanup:
   - keep primary connected component
   - remove sparse floaters
   - preserve thin object parts
3. Add benchmark scenes:
   - toy/object turntable
   - indoor object
   - outdoor object
   - public datasets when licenses permit
4. Add baseline runners:
   - this gsplat trainer
   - Nerfstudio Splatfacto
   - OpenSplat
5. Add GitHub-ready polish:
   - one-command demo
   - screenshots
   - troubleshooting guide
   - CI smoke tests for import/export utilities
