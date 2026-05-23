"""Export 3DGS to .splat format for WebGL viewing.

The .splat format (from antimatter15/splat) is a compact binary format:
  - float32 x, y, z        (12 bytes)
  - float32 scale_x,y,z     (12 bytes)
  - uint8   r, g, b, a      (4 bytes)
  - uint8   rot_0,1,2,3     (4 bytes)
  Total: 32 bytes per Gaussian

This is ~6x smaller than standard PLY and loads fast on mobile.
"""

import struct
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def quat_to_uint8(quat: np.ndarray) -> np.ndarray:
    """Convert float32 quaternion [w,x,y,z] to uint8 [0,255] range."""
    # Map [-1, 1] → [0, 255]
    quat = np.clip(quat, -1.0, 1.0)
    return ((quat + 1.0) / 2.0 * 255.0).astype(np.uint8)


def sh_to_rgb(sh_coeffs: np.ndarray, sh_degree: int = 3) -> np.ndarray:
    """Evaluate spherical harmonics at zero direction to get base RGB color.

    For the .splat format we only need the base color (DC component),
    which is sh0 / sqrt(4*pi). View-dependent SH is handled by adjusting
    opacity instead in the WebGL viewer.

    Args:
        sh_coeffs: SH coefficients [N, (degree+1)^2, 3] or [N, 3]
        sh_degree: SH degree (0-3)

    Returns:
        RGB colors [N, 3] in [0, 1]
    """
    if sh_coeffs.ndim == 2:
        # Already [N, 3]
        pass
    elif sh_coeffs.ndim == 3:
        # [N, C, 3] — take DC (index 0)
        sh_coeffs = sh_coeffs[:, 0, :]

    # Convert SH DC to RGB. 3DGS PLY stores the coefficient before the C0 basis.
    C0 = 0.28209479177387814
    rgb = 0.5 + C0 * sh_coeffs
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def compress_sort_indices(means: np.ndarray, cam_center: np.ndarray) -> np.ndarray:
    """Compute depth-sort indices for correct alpha blending.

    Sorts by distance from camera center (far to near for back-to-front).
    """
    depths = np.sum((means - cam_center) ** 2, axis=1)
    return np.argsort(-depths)  # descending = far first


class SplatExporter:
    """Export 3DGS checkpoint to .splat format.

    Usage:
        exporter = SplatExporter("checkpoint.pt")
        exporter.export("output.splat", sort_by_depth=True)
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

    def export(self, output_path: str,
               sort_by_depth: bool = True,
               opacity_threshold: float = 0.0,
               max_gaussians: Optional[int] = None):
        """Export checkpoint to .splat file.

        Args:
            output_path: Output .splat file path
            sort_by_depth: Sort Gaussians for correct alpha blending
            opacity_threshold: Filter out low-opacity Gaussians
            max_gaussians: Cap the number of Gaussians (for mobile optimization)
        """
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        splats = ckpt["splats"]

        means = splats["means"].numpy()
        opacities = torch.sigmoid(splats["opacities"]).numpy().squeeze(-1)
        scales = torch.exp(splats["scales"]).numpy()
        quats = splats["quats"].numpy()
        quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)

        # Get SH base color (handle both sh0 and sh_colors keys)
        sh0 = None
        for key in ["sh0", "sh_colors"]:
            if key in splats:
                sh0 = splats[key].numpy()
                break
        if sh0 is not None and sh0.ndim == 3:
            sh0 = sh0[:, 0, :] if sh0.shape[1] > 3 else sh0[:, 0, :]
        if sh0 is None:
            sh0 = np.ones((len(means), 3)) * 0.5

        rgb = sh_to_rgb(sh0)

        # Filter by opacity
        if opacity_threshold > 0:
            mask = opacities > opacity_threshold
            means = means[mask]
            scales = scales[mask]
            quats = quats[mask]
            rgb = rgb[mask]
            opacities = opacities[mask]

        # Cap count
        if max_gaussians and len(means) > max_gaussians:
            # Keep highest-opacity Gaussians
            indices = np.argsort(-opacities)[:max_gaussians]
            means = means[indices]
            scales = scales[indices]
            quats = quats[indices]
            rgb = rgb[indices]
            opacities = opacities[indices]

        N = len(means)
        print(f"[SplatExport] Exporting {N} Gaussians to .splat")

        # Sort by depth for correct blending
        if sort_by_depth:
            cam_center = np.mean(means, axis=0) + np.array([0, 0, 3.0])
            sort_idx = compress_sort_indices(means, cam_center)
            means = means[sort_idx]
            scales = scales[sort_idx]
            quats = quats[sort_idx]
            rgb = rgb[sort_idx]
            opacities = opacities[sort_idx]

        # Compress quaternions to uint8
        quat_uint8 = quat_to_uint8(quats)

        # Map opacity from sigmoid space to alpha [0, 255]
        alpha_uint8 = (np.clip(opacities, 0, 1) * 255).astype(np.uint8)

        # RGBA colors
        rgba = np.column_stack([
            (rgb[:, 0] * 255).astype(np.uint8),
            (rgb[:, 1] * 255).astype(np.uint8),
            (rgb[:, 2] * 255).astype(np.uint8),
            alpha_uint8,
        ])

        # Write binary .splat file
        with open(output_path, "wb") as f:
            for i in range(N):
                f.write(struct.pack("<fff", *means[i]))
                f.write(struct.pack("<fff", *scales[i]))
                f.write(struct.pack("<BBBB", *rgba[i]))
                f.write(struct.pack("<BBBB", *quat_uint8[i]))

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"[SplatExport] Written: {output_path} ({size_mb:.1f} MB, "
              f"{32 * N / 1024:.0f} KB raw)")
        return output_path


def export_from_ply(ply_path: str, splat_path: str,
                    sort_by_depth: bool = True,
                    max_gaussians: Optional[int] = None):
    """Convert standard PLY to .splat format.

    Supports both standard 3DGS PLY and the Inria format.
    """
    # Read PLY binary
    with open(ply_path, "rb") as f:
        lines = []
        while True:
            line = f.readline().decode("utf-8").strip()
            lines.append(line)
            if line == "end_header":
                break

        # Parse header
        properties = []
        vertex_count = 0
        for line in lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property float") or line.startswith("property uchar"):
                parts = line.split()
                properties.append((parts[1], parts[2]))

        # Read data (file is still open within the with block)
        dtype_map = {"float": np.float32, "uchar": np.uint8}
        dtype = np.dtype([(p[1], dtype_map[p[0]]) for p in properties])
        data = np.fromfile(f, dtype=dtype, count=vertex_count)

    # Extract fields
    x, y, z = data["x"], data["y"], data["z"]
    means = np.column_stack([x, y, z])

    # Try different PLY formats
    if "scale_0" in data.dtype.names:
        scales = np.column_stack([
            data["scale_0"], data["scale_1"], data["scale_2"]
        ])
        if np.nanmedian(scales) < -0.25:
            scales = np.exp(scales)
    else:
        scales = np.ones((vertex_count, 3)) * 0.01

    if "rot_0" in data.dtype.names:
        quats = np.column_stack([
            data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]
        ])
    else:
        quats = np.zeros((vertex_count, 4))
        quats[:, 0] = 1.0

    if "f_dc_0" in data.dtype.names:
        rgb = np.column_stack([
            data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]
        ])
        rgb = sh_to_rgb(rgb)
        rgb = np.clip(rgb, 0, 1)
    elif "red" in data.dtype.names:
        rgb = np.column_stack([
            data["red"], data["green"], data["blue"]
        ]) / 255.0
    else:
        rgb = np.ones((vertex_count, 3)) * 0.5

    if "opacity" in data.dtype.names:
        opacities = data["opacity"]
        if opacities.dtype == np.float32 and (opacities.min() < 0.0 or opacities.max() > 1.0):
            opacities = 1.0 / (1.0 + np.exp(-opacities))
    else:
        opacities = np.ones(vertex_count) * 0.5

    # Cap
    if max_gaussians and vertex_count > max_gaussians:
        indices = np.argsort(-opacities)[:max_gaussians]
        means = means[indices]
        scales = scales[indices]
        quats = quats[indices]
        rgb = rgb[indices]
        opacities = opacities[indices]

    # Sort
    if sort_by_depth:
        cam_center = np.mean(means, axis=0) + np.array([0, 0, 3.0])
        depths = np.sum((means - cam_center) ** 2, axis=1)
        sort_idx = np.argsort(-depths)
        means = means[sort_idx]
        scales = scales[sort_idx]
        quats = quats[sort_idx]
        rgb = rgb[sort_idx]
        opacities = opacities[sort_idx]

    N = len(means)
    quat_uint8 = quat_to_uint8(quats)
    alpha_uint8 = (np.clip(opacities, 0, 1) * 255).astype(np.uint8)
    rgba = np.column_stack([
        (rgb[:, 0] * 255).astype(np.uint8),
        (rgb[:, 1] * 255).astype(np.uint8),
        (rgb[:, 2] * 255).astype(np.uint8),
        alpha_uint8,
    ])

    with open(splat_path, "wb") as f:
        for i in range(N):
            f.write(struct.pack("<fff", *means[i]))
            f.write(struct.pack("<fff", *scales[i]))
            f.write(struct.pack("<BBBB", *rgba[i]))
            f.write(struct.pack("<BBBB", *quat_uint8[i]))

    size_mb = Path(splat_path).stat().st_size / (1024 * 1024)
    print(f"[SplatExport] {N} Gaussians → {splat_path} ({size_mb:.1f} MB)")
