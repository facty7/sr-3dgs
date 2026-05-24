"""Export utilities for SR-3DGS results.

Supports:
- PLY export (standard 3DGS format, compatible with all viewers)
- Trajectory video rendering
- WebGL-compatible splat export (for online delivery to clients)
- Colmap-compatible format export
"""

import struct
import json
from pathlib import Path
from typing import Optional, List

import numpy as np

from .utils import ensure_dir


def export_to_ply(checkpoint_path: str, output_path: str,
                  opacity_threshold: float = 0.0):
    """Export a checkpoint to PLY format.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_path: Output PLY file path
        opacity_threshold: Minimum opacity to include (0 = export all)
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    splats = ckpt["splats"]

    means = splats["means"].numpy()
    opacities = torch.sigmoid(splats["opacities"]).numpy()
    scales = torch.exp(splats["scales"]).numpy()
    quats = splats["quats"].numpy()
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)

    sh0 = splats.get("sh0", None)
    if sh0 is not None:
        sh0 = sh0.numpy()
        if sh0.ndim == 3:
            sh0 = sh0[:, 0, :]

    # Apply opacity filter
    if opacity_threshold > 0:
        mask = opacities.squeeze(-1) > opacity_threshold
        means = means[mask]
        opacities = opacities[mask]
        scales = scales[mask]
        quats = quats[mask]
        if sh0 is not None:
            sh0 = sh0[mask]

    N = len(means)
    if sh0 is None:
        sh0 = np.ones((N, 3)) * 0.5

    with open(output_path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
        f.write(b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n")
        for i in range(45):  # 15 * 3 for SH rest
            f.write(f"property float f_rest_{i}\n".encode())
        f.write(b"property float opacity\n")
        f.write(b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        f.write(b"property float rot_0\nproperty float rot_1\n"
                b"property float rot_2\nproperty float rot_3\n")
        f.write(b"end_header\n")

        for i in range(N):
            f.write(struct.pack("<fff", *means[i]))
            f.write(struct.pack("<fff", 1.0, 0.0, 0.0))
            f.write(struct.pack("<fff", *sh0[i]))
            f.write(struct.pack("<" + "f" * 45, *([0.0] * 45)))
            f.write(struct.pack("<f", float(opacities[i].squeeze())))
            f.write(struct.pack("<fff", *scales[i]))
            f.write(struct.pack("<ffff", *quats[i]))

    print(f"Exported {N} Gaussians to {output_path}")


def render_trajectory_video(checkpoint_path: str, output_path: str,
                            trajectory: str = "spiral",
                            num_frames: int = 120,
                            resolution: tuple = (1920, 1080),
                            fps: int = 30):
    """Render a camera trajectory video from a checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint
        output_path: Output MP4 path
        trajectory: 'spiral', 'circle', or 'interpolated'
        num_frames: Number of frames
        resolution: (width, height)
        fps: Frames per second
    """
    try:
        from .backends import ensure_local_gsplat
        ensure_local_gsplat()
        from gsplat.rendering import rasterization
    except ImportError:
        raise ImportError("gsplat required for rendering. pip install gsplat")

    import torch
    import imageio

    ckpt = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    splats = ckpt["splats"]
    device = splats["means"].device

    W, H = resolution
    K = torch.tensor([
        [W / 2, 0, W / 2],
        [0, H / 2, H / 2],
        [0, 0, 1],
    ], device=device).float()

    frames = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        # Simple circular/spiral orbit
        if trajectory == "spiral":
            radius = 3.0 + 0.5 * np.sin(angle * 0.5)
        else:
            radius = 3.0

        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = 0.0

        # Look-at camera
        forward = np.array([-cam_x, -cam_y, 0.0])
        forward = forward / np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        camtoworld = np.eye(4)
        camtoworld[:3, 0] = right
        camtoworld[:3, 1] = up
        camtoworld[:3, 2] = forward
        camtoworld[:3, 3] = [cam_x, cam_y, cam_z]
        camtoworld = torch.from_numpy(camtoworld).float().to(device)

        with torch.no_grad():
            viewmat = torch.linalg.inv(camtoworld)
            # Ensure colors in SH format [N, 1, 3]
            sh0 = splats["sh0"]
            if sh0.dim() == 2:
                sh0 = sh0.unsqueeze(1)  # [N, 3] -> [N, 1, 3]
            opas = torch.sigmoid(splats["opacities"])
            if opas.dim() == 2:
                opas = opas.squeeze(-1)
            render_col, _, _ = rasterization(
                means=splats["means"],
                quats=splats["quats"] / splats["quats"].norm(dim=-1, keepdim=True),
                scales=torch.exp(splats["scales"]),
                opacities=opas,
                colors=sh0,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=W,
                height=H,
                sh_degree=0,
                camera_model="pinhole",
                packed=False,
            )

        img = render_col[0].cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(img)

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Rendered {num_frames} frames to {output_path}")
