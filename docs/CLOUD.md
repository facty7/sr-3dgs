# Cloud GPU Notes

The pipeline can run on local Linux/WSL machines or cloud GPU instances. Choose
the environment based on scene size, GPU memory, and how much setup work you
want to manage.

## Common Options

- Local workstation: best for small experiments and reproducible development.
- AutoDL, Vast.ai, RunPod, or similar GPU rentals: useful for larger scenes or
  faster iteration.
- Modal or other serverless GPU platforms: useful for custom automation, but
  requires packaging the workflow carefully.

## Practical Advice

- Use persistent storage for `workspace_video/` and `output/`.
- Do not put large generated files in git.
- Run `scripts/preflight_heavy.py` before expensive conversion steps.
- Download or publish only final deliverables from `output/<scene>/`.
- Record the GPU type, driver, CUDA version, and command line when reporting
  issues.

Cloud pricing and GPU availability change frequently, so this repository avoids
hard-coded cost recommendations.
