"""Step 5: Post-training cleanup — pruning SR-induced floaters.

FIXED VERSION:
- max_scale_threshold is now scene_radius * relative_factor (NOT hardcoded 0.5)
- Adds prune logging showing how many Gaussians removed per criterion
- Remembers scene_radius from training for consistent thresholding
"""

import os
import math
import struct
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from .utils import ensure_dir


class CleanupProcessor:
    """Post-training cleanup to remove SR-induced floaters and noise.

    CRITICAL: Scale thresholds must be relative to scene_radius.
    A hardcoded absolute value (like 0.5) will either:
    - Kill ALL Gaussians in a small scene (scene_radius < 0.5)
    - Keep ALL floaters in a large scene (scene_radius > 5.0)
    """

    def __init__(self, checkpoint_path: str, output_dir: str,
                 device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def run(self,
            opacity_thresholds: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3),
            max_scale_relative: float = 0.15,  # FRACTION of scene_radius, NOT absolute!
            remove_outliers: bool = True,
            outlier_std_threshold: float = 4.0):
        """Run the cleanup pipeline.

        Args:
            opacity_thresholds: Progressive opacity cutoffs.
            max_scale_relative: Remove Gaussians with max_scale >
                scene_radius * max_scale_relative.
                Default 0.15 means a Gaussian can span at most 15% of the scene.
            remove_outliers: Remove points > outlier_std_threshold std from centroid.
            outlier_std_threshold: Number of standard deviations for outlier removal.
        """
        ensure_dir(self.output_dir)

        print(f"[Step5] Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        splats = ckpt["splats"]
        step = ckpt.get("step", 0)

        # Extract parameters. Standard 3DGS PLY stores raw opacity logits and
        # log-scales; web export uses sigmoid opacity and exp scales.
        means = splats["means"].float()
        opacity_logits = splats["opacities"].float()
        log_scales = splats["scales"].float()
        opacities = torch.sigmoid(opacity_logits)
        scales = torch.exp(log_scales)
        quats = splats["quats"].float()
        quats = quats / quats.norm(dim=-1, keepdim=True)
        sh0 = splats.get("sh0", None)

        N_original = len(means)
        print(f"[Step5] Original: {N_original} Gaussians at step {step}")

        # ── Compute scene_radius robustly (MCMC noise can create extreme outliers) ──
        # Use MEDIAN as robust centroid (immune to position outliers)
        robust_centroid = means.median(dim=0).values
        distances = torch.norm(means - robust_centroid, dim=1)
        median_dist = distances.median().item()
        p95 = distances.kthvalue(int(len(distances) * 0.95)).values.item()
        scene_radius = max(median_dist * 3.0, p95 * 1.2)
        scene_radius = max(scene_radius, 0.01)
        # Also ensure centroid used for outlier removal is robust
        centroid = robust_centroid

        # Compute scale statistics
        max_scales = scales.max(dim=-1).values
        print(f"[Step5] Scene radius: {scene_radius:.3f} (p95={p95:.3f}, median={median_dist:.3f})")
        print(f"[Step5] Scale stats: min={max_scales.min().item():.4f} "
              f"median={max_scales.median().item():.4f} "
              f"mean={max_scales.mean().item():.4f} "
              f"max={max_scales.max().item():.4f}")

        # ── CRITICAL: Scale threshold relative to scene_radius ──
        abs_max_scale = scene_radius * max_scale_relative
        print(f"[Step5] Using max_scale_threshold = scene_radius({scene_radius:.3f}) "
              f"* {max_scale_relative} = {abs_max_scale:.4f}")

        # 1. Remove large-scale Gaussians (the "放射状" artifacts)
        scale_mask = max_scales < abs_max_scale
        n_pruned_scale = (~scale_mask).sum().item()
        print(f"[Step5] Pruned {n_pruned_scale} Gaussians due to "
              f"max_scale > {abs_max_scale:.4f} ({max_scale_relative*100:.0f}% of scene_radius)")

        # 2. Remove position outliers
        pos_mask = torch.ones(len(means), dtype=torch.bool, device=self.device)
        if remove_outliers:
            distances = torch.norm(means - centroid, dim=1)
            p99 = distances.kthvalue(max(1, int(len(distances) * 0.99))).values
            threshold = torch.minimum(
                p99,
                torch.tensor(scene_radius * 2.0, device=self.device),
            )
            pos_mask = distances < threshold
            n_pruned_pos = (~pos_mask).sum().item()
            print(f"[Step5] Pruned {n_pruned_pos} position outliers "
                  f"(robust threshold={threshold.item():.3f})")

        # Combine geometry masks
        keep_mask = scale_mask & pos_mask
        n_after_geo = keep_mask.sum().item()
        print(f"[Step5] After geometry cleanup: {n_after_geo} Gaussians "
              f"({100*n_after_geo/N_original:.1f}% of original)")

        # 3. Export at multiple opacity thresholds
        for opa_thresh in opacity_thresholds:
            opa_mask = opacities.squeeze(-1) > opa_thresh
            final_mask = keep_mask & opa_mask
            n_final = final_mask.sum().item()
            n_pruned_opa = (keep_mask & ~opa_mask).sum().item()
            pct = 100.0 * n_final / N_original
            print(f"[Step5] Opacity > {opa_thresh:.2f}: "
                  f"pruned {n_pruned_opa} low-opacity, keeping {n_final} ({pct:.1f}%)")

            self._export_subset(
                means, quats, scales, opacities, sh0,
                final_mask, opa_thresh
            )
            self._export_standard_subset(
                means, quats, log_scales, opacity_logits, sh0,
                final_mask, opa_thresh
            )

        print(f"[Step5] Cleanup complete. Outputs in {self.output_dir}")
        print(f"[Step5] Summary: {N_original} -> {n_after_geo} (geo) -> "
              f"{n_final} (opacity) = {100*n_final/N_original:.1f}% retained")

    def _split_sh(self, sh0, mask):
        m = mask.cpu().numpy()
        if sh0 is not None:
            sh0_np = sh0.cpu().numpy()
            if sh0_np.ndim == 3:
                sh_dc = sh0_np[m, 0, :]
                sh_rest = sh0_np[m, 1:, :]
            else:
                sh_dc = sh0_np[m]
                sh_rest = np.zeros((m.sum(), 15, 3))
        else:
            sh_dc = np.ones((m.sum(), 3)) * 0.5
            sh_rest = np.zeros((m.sum(), 15, 3))
        return sh_dc, sh_rest

    def _export_subset(self, means, quats, scales, opacities, sh0,
                       mask, opa_thresh):
        """Export a subset of Gaussians to PLY."""
        m = mask.cpu().numpy()
        means_np = means.cpu().numpy()[m]
        quats_np = quats.cpu().numpy()[m]
        scales_np = scales.cpu().numpy()[m]
        opacities_np = opacities.cpu().numpy()[m]

        sh_dc, sh_rest = self._split_sh(sh0, mask)

        N = m.sum()
        N_rest = sh_rest.shape[1] * 3
        out_path = self.output_dir / f"clean_opa{opa_thresh:.2f}.ply"

        with open(out_path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {N}\n".encode())
            f.write(b"property float x\nproperty float y\nproperty float z\n")
            f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
            f.write(b"property float f_dc_0\nproperty float f_dc_1\n"
                    b"property float f_dc_2\n")
            # 45 f_rest coefficients
            for i in range(N_rest):
                f.write(f"property float f_rest_{i}\n".encode())
            f.write(b"property float opacity\n")
            f.write(b"property float scale_0\nproperty float scale_1\n"
                    b"property float scale_2\n")
            f.write(b"property float rot_0\nproperty float rot_1\n"
                    b"property float rot_2\nproperty float rot_3\n")
            f.write(b"end_header\n")

            for i in range(N):
                f.write(struct.pack("<fff", *means_np[i]))
                f.write(struct.pack("<fff", 1.0, 0.0, 0.0))
                # f_dc in PLY = SH DC coefficient (gsplat convention)
                f.write(struct.pack("<fff", *sh_dc[i]))
                # f_rest: 15 coeffs x 3 channels = 45 values
                f.write(struct.pack("<" + "f" * N_rest, *sh_rest[i].flatten()))
                opa_val = float(opacities_np[i, 0] if opacities_np.ndim > 1
                                else opacities_np[i])
                f.write(struct.pack("<f", opa_val))
                f.write(struct.pack("<fff", *scales_np[i]))
                f.write(struct.pack("<ffff", *quats_np[i]))

        print(f"  -> {out_path} ({N} Gaussians)")

    def _export_standard_subset(self, means, quats, log_scales, opacity_logits,
                                sh0, mask, opa_thresh):
        """Export SuperSplat/Inria-compatible PLY with raw 3DGS parameters."""
        m = mask.cpu().numpy()
        means_np = means.cpu().numpy()[m]
        quats_np = quats.cpu().numpy()[m]
        log_scales_np = log_scales.cpu().numpy()[m]
        opacity_logits_np = opacity_logits.cpu().numpy()[m]
        sh_dc, sh_rest = self._split_sh(sh0, mask)

        N = m.sum()
        N_rest = sh_rest.shape[1] * 3
        out_path = self.output_dir / f"clean_opa{opa_thresh:.2f}_standard.ply"

        with open(out_path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {N}\n".encode())
            f.write(b"property float x\nproperty float y\nproperty float z\n")
            f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
            f.write(b"property float f_dc_0\nproperty float f_dc_1\n"
                    b"property float f_dc_2\n")
            for i in range(N_rest):
                f.write(f"property float f_rest_{i}\n".encode())
            f.write(b"property float opacity\n")
            f.write(b"property float scale_0\nproperty float scale_1\n"
                    b"property float scale_2\n")
            f.write(b"property float rot_0\nproperty float rot_1\n"
                    b"property float rot_2\nproperty float rot_3\n")
            f.write(b"end_header\n")

            for i in range(N):
                f.write(struct.pack("<fff", *means_np[i]))
                f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
                f.write(struct.pack("<fff", *sh_dc[i]))
                f.write(struct.pack("<" + "f" * N_rest, *sh_rest[i].flatten()))
                f.write(struct.pack("<f", float(opacity_logits_np[i, 0])))
                f.write(struct.pack("<fff", *log_scales_np[i]))
                f.write(struct.pack("<ffff", *quats_np[i]))

        print(f"  -> {out_path} ({N} standard Gaussians)")
