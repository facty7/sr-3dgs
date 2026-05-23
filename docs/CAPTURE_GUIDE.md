# Capture Guide

Good reconstruction starts before the pipeline. This guide describes captures
that the current workflow can handle best.

## Best Case

- One main object.
- Object stays still.
- Camera moves around the object, not just side to side.
- 40-180 useful views after extraction.
- Matte or textured surface.
- Stable lighting.
- Background is visually different from the object.

## Phone Video

Recommended capture:

- Record 20-60 seconds.
- Move in a slow circle around the object.
- Capture three height bands when possible: low, middle, high.
- Keep the object centered and fully visible.
- Avoid digital zoom.
- Avoid fast motion blur.
- Avoid reflective, transparent, or very thin objects for early tests.

Useful command shape:

```bash
python scripts/run_video_pipeline.py \
  --video input_videos/object.mp4 \
  --output_name object \
  --projection perspective \
  --preset standard \
  --object_mask auto \
  --cluster_clean
```

If the object is small in the frame, add an object crop:

```bash
--object_bbox left,top,right,bottom
```

The video pipeline writes lightweight input reports under
`workspace_video/<scene>/reports/` after frame extraction, and again after
object mask generation when `--object_mask auto` is enabled. Read these before
spending time on longer training runs:

- `input_quality_frames.html`
- `input_quality_object.html`

## 360 Video

For equirectangular 360 videos, the pipeline converts selected frames into
perspective cube faces before COLMAP:

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

Current limitations:

- Cube-face extraction is a practical bridge, not full spherical SfM.
- Very close objects can distort across face boundaries.
- Top/down faces are disabled by default because they often add floor/ceiling
  clutter for object captures.

## Image Folders

Use image folders when you already have clean stills:

- Keep EXIF if possible.
- Remove blurry or duplicate images.
- Include around-object coverage.
- Avoid mixing unrelated backgrounds or object states.

## Input Quality Check

Before spending time on a long training run, create a lightweight report for
the extracted frames and masks:

```bash
python scripts/assess_scene_inputs.py workspace_video/object \
  --report workspace_video/object/reports/input_quality.json \
  --html workspace_video/object/reports/input_quality.html
```

The report checks frame count, blur, near-duplicate views, large frame jumps,
mask foreground size, and whether masks cut into the image edges. It is meant
to answer a practical question: should you reshoot or adjust extraction/masks
before tuning training settings?

## Common Failure Modes

- Too few angles: output looks flat or incomplete.
- Motion blur: COLMAP misses matches.
- Background moves: background becomes part of the object.
- Object moves/deforms: reconstruction becomes fuzzy.
- Shiny/transparent surfaces: splats float or smear.
- Object and background have similar colors: masks leak.

## Minimum Acceptance

For a useful object demo, expect:

- recognisable object silhouette
- compact SOG under the benchmark gate
- standard PLY that opens in SuperSplat
- limited detached floaters after auto cleanup
- final files under `output/<scene>`

The benchmark and smoke checks verify delivery health, not artistic quality.
Visual review is still required before calling a scene production-ready.
