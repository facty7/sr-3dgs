# Project Positioning

This project is not a new 3D Gaussian Splatting research method. It is an
object-focused delivery pipeline that connects capture, reconstruction,
cleanup, web packaging, and release checks.

## Existing Strong Projects

- Nerfstudio / Splatfacto: strong training and research workflow for Gaussian
  Splatting, including variants for unconstrained photo collections.
- gsplat: open-source Gaussian Splatting rasterization and training
  primitives.
- PlayCanvas SuperSplat: browser-based editing, cleanup, publishing, and SOG
  delivery tooling.
- PlayCanvas SOG / splat-transform: compact web/mobile Gaussian Splat delivery
  format and converter.
- Postshot, Splatica, and similar products: polished app/cloud workflows for
  capture-to-splat creation.

## What This Project Adds

The intended value is the full object-delivery workflow:

- phone video or image folders as practical inputs
- optional 360/equirectangular frame extraction into perspective cube faces
- COLMAP alignment fixes and reproducible training entry points
- object crop and mask-aware training hooks
- automatic Gaussian cleanup for floaters, detached clusters, and haze splats
- flat `output/<scene>` delivery folders
- SOG for lightweight web/mobile viewing
- standard PLY for SuperSplat and professional tools
- smoke, benchmark, CI, and release-readiness checks

## What It Does Not Claim

- It does not guarantee high-quality reconstruction from low-coverage, blurred, or unstable captures.
- It does not replace Nerfstudio, gsplat, SuperSplat, or commercial products.
- It does not solve segmentation perfectly yet.
- It does not make browser/WebGL render QA safe to run by default on every
  workstation.

## Current Quality Bar

The toy demo is a useful proof of pipeline closure, not a final benchmark suite.
It proves that the project can produce a web SOG, standard PLY, flat delivery
folder, diagnostics, and automated smoke checks. More public scenes are needed
before claiming broad quality.

Current gates live in `configs/benchmark_outputs.json`:

- SOG <= 12 MB
- PLY point count >= 120k
- output score >= 85

## Roadmap To Strong Open Source

1. Add at least three public object-scene tests to `configs/benchmark_outputs.json`.
2. Improve foreground segmentation beyond the current fast/rembg options.
3. Add a safe, optional visual QA mode that can run on a dedicated machine.
4. Document recommended capture patterns for phones and 360 cameras.
5. Publish demo outputs as release artifacts, not as git-tracked files.
