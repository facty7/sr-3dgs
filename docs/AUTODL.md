# AutoDL Notes

AutoDL can be useful when a local workstation does not have enough GPU memory.
This document is intentionally brief and vendor-neutral; prices, regions, and
available GPU types change often.

## Suggested Workflow

1. Create a GPU instance with a persistent data volume.
2. Clone the repository onto the persistent volume.
3. Install dependencies inside a virtual environment or conda environment.
4. Upload input media into an ignored local folder such as `input_videos/`.
5. Run the same commands documented in the main README.
6. Download `output/<scene>/` or attach it as a release artifact.
7. Stop the instance when idle.

## Example

```bash
git clone <repo-url> sr_3dgs
cd sr_3dgs
python -m venv .venv
source .venv/bin/activate
pip install -e ".[training]"
pip install -r requirements.txt

python scripts/run_video_pipeline.py \
  --video input_videos/object.mp4 \
  --output_name object \
  --projection perspective \
  --preset standard \
  --object_mask auto \
  --cluster_clean
```

Keep generated workspaces on persistent storage when stop/resume workflows are
expected.
