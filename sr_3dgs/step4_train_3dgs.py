"""Step 4: Train 3DGS using gsplat strategies."""

import os, math, time, struct
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ensure_dir
from .backends import ensure_local_gsplat


@dataclass
class SRTrainConfig:
    data_dir: str = ""
    result_dir: str = "output/sr_3dgs"

    # Training schedule
    max_steps: int = 15_000
    eval_steps: int = 2_000
    save_steps: int = 5_000
    warmup_steps: int = 500
    warmup_lr_factor: float = 0.1
    sh_degree: int = 2
    sh_degree_interval: int = 500

    # Initialization
    init_opa: float = 0.5
    init_scale_bias: float = -3.0
    init_scale_multiplier: float = 1.0
    max_init_scale_fraction: float = 0.10
    max_train_scale_fraction: float = 0.10

    # gsplat strategy. "default" is the official densify/prune path; "mcmc"
    # is kept as a bounded fallback for tight VRAM/mobile budgets.
    strategy: str = "default"

    # MCMCStrategy
    cap_max: int = 200_000
    noise_lr: float = 1e5

    # DefaultStrategy
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    reset_every: int = 3000
    absgrad: bool = True

    # Shared refinement schedule
    refine_start_iter: int = 500
    refine_stop_iter: int = 12_000
    refine_every: int = 100
    min_opacity: float = 0.005

    # Regularization
    opacity_reg: float = 0.01
    scale_reg: float = 0.05
    ssim_lambda: float = 0.2
    mask_dir: str = ""
    mask_foreground_weight: float = 1.0
    mask_background_weight: float = 0.05
    mask_alpha_reg: float = 0.05

    # Optimizer
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh: float = 2.5e-3

    # Rendering
    camera_model: str = "pinhole"
    background_color: tuple = (0.5, 0.5, 0.5)
    device: str = "cuda"
    seed: int = 42
    data_factor: int = 1
    max_render_dim: int = 1600


class SR3DGSTrainer:

    def __init__(self, config: SRTrainConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)

    def run(self):
        cfg = self.config
        ensure_dir(cfg.result_dir)
        data = self._load_data()
        self._train(data)

    def _load_data(self):
        cfg = self.config
        data_path = Path(cfg.data_dir) / "scene_data.npz"
        loaded = np.load(data_path, allow_pickle=True)
        transforms = loaded["transforms"].tolist()
        points_xyz = loaded["points_xyz"]
        points_rgb = loaded["points_rgb"]
        print(f"[Step4] Loaded {len(transforms)} images, {len(points_xyz)} points.")
        return {
            "transforms": transforms,
            "points_xyz": points_xyz,
            "points_rgb": points_rgb,
        }

    def _init_gaussians(self, points_xyz, points_rgb, scene_scale):
        """Initialize Gaussians with scene-relative scale clamping."""
        cfg = self.config
        N = len(points_xyz)
        device = self.device
        means = nn.Parameter(torch.from_numpy(points_xyz).float().to(device))
        quats = nn.Parameter(torch.zeros(N, 4, device=device))
        quats.data[:, 0] = 1.0
        if N > 1:
            from sklearn.neighbors import NearestNeighbors
            pts = points_xyz
            if len(pts) > 10000:
                idx = np.random.choice(len(pts), 10000, replace=False)
                pts_sample = pts[idx]
            else:
                pts_sample = pts
            nbrs = NearestNeighbors(n_neighbors=min(4, len(pts_sample)))
            nbrs.fit(pts_sample)
            dists, _ = nbrs.kneighbors(pts)
            max_init_scale = scene_scale * cfg.max_init_scale_fraction
            avg_dists = np.clip(dists[:, 1:].mean(axis=1),
                                scene_scale * 0.0001,
                                max_init_scale)
            avg_dists = avg_dists * cfg.init_scale_multiplier
            scales = torch.log(torch.from_numpy(avg_dists).float().to(device).unsqueeze(-1).repeat(1, 3))
            scales = nn.Parameter(scales)
            print(f"[Step4] Init scales: max={max_init_scale:.3f}, "
                  f"mult={cfg.init_scale_multiplier:.2f}, N={N}")
        else:
            scales = nn.Parameter(torch.full((N, 3), cfg.init_scale_bias, device=device))
        init_logit = math.log(cfg.init_opa / (1.0 - cfg.init_opa))
        opacities = nn.Parameter(torch.full((N, 1), init_logit, device=device))
        sh_dim = (cfg.sh_degree + 1) ** 2
        sh_colors = nn.Parameter(torch.zeros(N, sh_dim, 3, device=device))
        C0 = 0.28209479177387814
        rgb_norm = torch.from_numpy(points_rgb).float().to(device) / 255.0
        sh_colors.data[:, 0, :] = (rgb_norm - 0.5) / C0
        return {"means": means, "quats": quats, "scales": scales,
                "opacities": opacities, "sh0": sh_colors}

    def _create_optimizers(self, gs, spatial_lr_scale):
        cfg = self.config
        return {
            "means": torch.optim.Adam([gs["means"]],
                                      lr=cfg.lr_means * spatial_lr_scale),
            "scales": torch.optim.Adam([gs["scales"]], lr=cfg.lr_scales),
            "quats": torch.optim.Adam([gs["quats"]], lr=cfg.lr_quats),
            "opacities": torch.optim.Adam([gs["opacities"]], lr=cfg.lr_opacities),
            "sh0": torch.optim.Adam([gs["sh0"]], lr=cfg.lr_sh),
        }

    def _clamp_scene_bounds(self, gs, scene_extent, reason="", verbose=True):
        """Keep MCMC relocations inside a sane object-space envelope."""
        cfg = self.config
        with torch.no_grad():
            max_allowed = scene_extent * cfg.max_train_scale_fraction
            scales_exp = torch.exp(gs["scales"])
            gs["scales"].data = torch.log(torch.clamp(scales_exp, max=max_allowed))

            pos_max = scene_extent * 2.0
            pos_norms = torch.norm(gs["means"], dim=-1)
            far = pos_norms > pos_max
            if far.any():
                gs["means"].data[far] *= (pos_max / pos_norms[far]).unsqueeze(-1)
                if reason and verbose:
                    print(f"[Step4] Clamped {far.sum().item()} position outliers ({reason})")

    def _init_strategy(self, scene_scale):
        cfg = self.config
        ensure_local_gsplat()
        from gsplat.strategy import DefaultStrategy, MCMCStrategy

        name = cfg.strategy.lower()
        if name == "default":
            strategy = DefaultStrategy(
                prune_opa=cfg.prune_opa,
                grow_grad2d=cfg.grow_grad2d,
                grow_scale3d=cfg.grow_scale3d,
                grow_scale2d=cfg.grow_scale2d,
                prune_scale3d=cfg.prune_scale3d,
                prune_scale2d=cfg.prune_scale2d,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                absgrad=cfg.absgrad,
                verbose=True,
            )
            state = strategy.initialize_state(scene_scale=scene_scale)
            print(f"[Step4] DefaultStrategy: refine {cfg.refine_start_iter}-{cfg.refine_stop_iter}, "
                  f"reset_every={cfg.reset_every}, absgrad={cfg.absgrad}")
            return name, strategy, state
        if name == "mcmc":
            strategy = MCMCStrategy(
                cap_max=cfg.cap_max,
                noise_lr=cfg.noise_lr,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                refine_every=cfg.refine_every,
                min_opacity=cfg.min_opacity,
                verbose=False,
            )
            state = strategy.initialize_state()
            print(f"[Step4] MCMCStrategy: cap_max={cfg.cap_max}, "
                  f"refine {cfg.refine_start_iter}-{cfg.refine_stop_iter}")
            return name, strategy, state
        raise ValueError(f"Unknown gsplat strategy: {cfg.strategy!r}. Use 'default' or 'mcmc'.")

    def _train(self, data):
        cfg = self.config
        transforms = data["transforms"]
        pts = data["points_xyz"]

        # Compute scene extent from camera centers
        cam_centers = np.array([np.array(t["camtoworld"])[:3, 3] for t in transforms])
        cam_center = cam_centers.mean(axis=0)
        cam_spread = np.linalg.norm(cam_centers - cam_center, axis=-1).max()
        scene_radius = max(cam_spread, 0.01)
        scene_extent = scene_radius * 2.0

        # Center scene at origin
        pts_mean_np = pts.mean(axis=0)
        pts_mean = torch.from_numpy(pts_mean_np).float().to(self.device)

        gs = self._init_gaussians(pts, data["points_rgb"], scene_radius)
        gs["means"].data -= pts_mean
        for t in transforms:
            ct = np.array(t["camtoworld"])
            ct[:3, 3] -= pts_mean_np
            t["camtoworld"] = ct.tolist()

        # Recompute after centering
        cam_centers = np.array([np.array(t["camtoworld"])[:3, 3] for t in transforms])
        cam_spread = np.linalg.norm(cam_centers - cam_centers.mean(axis=0), axis=-1).max()
        scene_radius_centered = max(cam_spread, 0.01)
        scene_extent_centered = scene_radius_centered * 2.0
        spatial_lr_scale = scene_extent_centered

        print(f"[Step4] Scene radius: {scene_radius_centered:.2f}, extent: {scene_extent_centered:.2f}")
        print(f"[Step4] spatial_lr_scale: {spatial_lr_scale:.3f} "
              f"(means LR: {cfg.lr_means * spatial_lr_scale:.2e})")

        opts = self._create_optimizers(gs, spatial_lr_scale)

        ensure_local_gsplat()
        from gsplat.rendering import rasterization
        from PIL import Image

        strategy_name, strategy, strategy_state = self._init_strategy(scene_extent_centered)
        strategy.check_sanity(gs, opts)

        current_sh_degree = 0
        N_initial = len(gs["means"])
        print(f"[Step4] Training {cfg.max_steps} steps")

        t0 = time.time()
        best_psnr = 0.0
        running_psnr = 0.0

        for step in range(1, cfg.max_steps + 1):
            # SH degree annealing
            target_sh = min(cfg.sh_degree,
                           max(0, (step - cfg.warmup_steps) // cfg.sh_degree_interval))
            if target_sh != current_sh_degree:
                current_sh_degree = target_sh
                if step > 1:
                    print(f"  Step {step}: SH degree -> {current_sh_degree}")

            # LR warmup
            lr_scale = cfg.warmup_lr_factor + \
                       (1.0 - cfg.warmup_lr_factor) * min(1.0, step / cfg.warmup_steps)

            # Random view with render resolution control
            idx = np.random.randint(0, len(transforms))
            view = transforms[idx]
            camtoworld = torch.tensor(view["camtoworld"], device=self.device).float()
            K = torch.tensor(view["K"], device=self.device).float()
            W, H = view["width"], view["height"]

            render_scale = 1.0
            if cfg.data_factor > 1:
                render_scale /= cfg.data_factor
            max_dim = max(W, H)
            if max_dim > cfg.max_render_dim:
                extra = (max_dim + cfg.max_render_dim - 1) // cfg.max_render_dim
                render_scale /= extra
            if render_scale < 1.0:
                W = max(1, int(W * render_scale))
                H = max(1, int(H * render_scale))
                K_scaled = K.clone()
                K_scaled[0, 0] *= render_scale
                K_scaled[1, 1] *= render_scale
                K_scaled[0, 2] *= render_scale
                K_scaled[1, 2] *= render_scale
                K = K_scaled

            gt_path = Path(cfg.data_dir) / "images" / view["image_name"]
            if not gt_path.exists():
                continue
            gt_img_full = np.array(Image.open(gt_path).convert("RGB")).astype(np.float32) / 255.0
            mask_img_full = None
            if cfg.mask_dir:
                mask_path = Path(cfg.mask_dir) / (Path(view["image_name"]).stem + ".png")
                if mask_path.exists():
                    mask_img_full = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0
            if render_scale < 1.0:
                import cv2
                gt_img_full = cv2.resize(gt_img_full, (W, H), interpolation=cv2.INTER_AREA)
                if mask_img_full is not None:
                    mask_img_full = cv2.resize(mask_img_full, (W, H), interpolation=cv2.INTER_AREA)
            gt_img = torch.from_numpy(gt_img_full).to(self.device)
            mask_img = None
            if mask_img_full is not None:
                mask_img = torch.from_numpy(mask_img_full).to(self.device).clamp(0.0, 1.0)

            # Rasterize with packed=False for VRAM efficiency
            viewmat = torch.linalg.inv(camtoworld)
            bg_color = torch.tensor(cfg.background_color, device=self.device).reshape(1, 3)
            render_col, render_alpha, info = rasterization(
                means=gs["means"],
                quats=gs["quats"] / gs["quats"].norm(dim=-1, keepdim=True),
                scales=torch.exp(gs["scales"]),
                opacities=torch.sigmoid(gs["opacities"]).squeeze(-1),
                colors=gs["sh0"],
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=W, height=H,
                sh_degree=current_sh_degree,
                camera_model=cfg.camera_model,
                backgrounds=bg_color,
                absgrad=cfg.absgrad if strategy_name == "default" else False,
                packed=False,
            )

            if strategy_name == "mcmc":
                strategy.step_pre_backward(gs, opts, strategy_state, step, info, cfg.noise_lr)
            else:
                strategy.step_pre_backward(gs, opts, strategy_state, step, info)

            # Loss: L1 + SSIM + regularization
            render_rgb = render_col[0]
            render_a = render_alpha[0].squeeze(-1)
            gt_rgb = gt_img[:H, :W]
            if mask_img is not None:
                m = mask_img[:H, :W].unsqueeze(-1)
                weights = m * cfg.mask_foreground_weight + (1.0 - m) * cfg.mask_background_weight
                l1_loss = (torch.abs(render_rgb - gt_rgb) * weights).sum() / weights.sum().clamp_min(1e-6)
            else:
                l1_loss = F.l1_loss(render_rgb, gt_rgb)
            loss = l1_loss * (1.0 - cfg.ssim_lambda)

            try:
                from torchmetrics.image import StructuralSimilarityIndexMeasure
                if mask_img is None:
                    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                    ssim_val = ssim(render_rgb.permute(2, 0, 1).unsqueeze(0),
                                    gt_rgb.permute(2, 0, 1).unsqueeze(0))
                    loss += (1.0 - ssim_val) * cfg.ssim_lambda
            except Exception:
                pass

            if mask_img is not None and cfg.mask_alpha_reg > 0:
                bg = (1.0 - mask_img[:H, :W])
                loss += cfg.mask_alpha_reg * (render_a * bg).sum() / bg.sum().clamp_min(1e-6)

            loss += cfg.opacity_reg * torch.abs(gs["opacities"]).mean()
            loss += cfg.scale_reg * torch.abs(torch.exp(gs["scales"])).mean()
            # L2 penalty on large scales: disproportionately punishes giant Gaussians
            max_per_gs = torch.exp(gs["scales"]).max(dim=-1).values
            large = max_per_gs > (scene_extent_centered * 0.05)
            if large.any():
                loss += 0.1 * (max_per_gs[large] ** 2).mean()

            loss.backward()

            if lr_scale != 1.0:
                for opt in opts.values():
                    for pg in opt.param_groups:
                        for p in pg["params"]:
                            if p.grad is not None:
                                p.grad.data.mul_(lr_scale)

            for opt in opts.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            if strategy_name == "mcmc":
                strategy.step_post_backward(gs, opts, strategy_state, step, info, cfg.noise_lr)
            else:
                strategy.step_post_backward(gs, opts, strategy_state, step, info, packed=False)

            # Hard scale and position clamps (prevent drift from MCMC noise)
            if step >= cfg.warmup_steps:
                self._clamp_scene_bounds(
                    gs,
                    scene_extent_centered,
                    reason=f"step {step}",
                    verbose=(step % 500 == 0),
                )

            if step % 500 == 0 or step == 1:
                psnr = -10.0 * math.log10(max(1e-10, (render_rgb - gt_rgb).pow(2).mean().item()))
                if step == 1: running_psnr = psnr
                else: running_psnr = 0.9 * running_psnr + 0.1 * psnr
                best_psnr = max(best_psnr, psnr)
                curr_scales = torch.exp(gs["scales"]).detach()
                max_sc = curr_scales.max(dim=-1).values
                cnt = len(gs["means"])
                print(f"  Step {step}/{cfg.max_steps} Loss: {loss.item():.4f} "
                      f"PSNR: {psnr:.2f} (avg: {running_psnr:.2f}) N: {cnt} "
                      f"| scale max={max_sc.max().item():.4f} "
                      f"p99={max_sc.kthvalue(int(cnt*0.99)).values.item():.4f}")

            if step % cfg.save_steps == 0:
                self._save_checkpoint(gs, step)
            if step % cfg.eval_steps == 0:
                self._eval(gs, transforms, step, current_sh_degree)

        elapsed = time.time() - t0
        self._clamp_scene_bounds(gs, scene_extent_centered, reason="final export")
        print(f"[Step4] Done in {elapsed:.0f}s ({elapsed/60:.1f}m). Best PSNR: {best_psnr:.2f}")
        print(f"[Step4] Gaussians: {N_initial} -> {len(gs['means'])}")
        self._save_checkpoint(gs, cfg.max_steps)
        self._export_ply(gs, cfg.max_steps)

    def _save_checkpoint(self, gs, step):
        ckpt_path = Path(self.config.result_dir) / f"checkpoint_step{step}.pt"
        state = {"step": step, "splats": {k: v.data.cpu() for k, v in gs.items()}}
        torch.save(state, ckpt_path)

    def _export_ply(self, gs, step):
        means = gs["means"].data.cpu().numpy()
        opas = torch.sigmoid(gs["opacities"].data).cpu().numpy().squeeze(-1)
        scales = torch.exp(gs["scales"].data).cpu().numpy()
        quats = (gs["quats"].data / gs["quats"].data.norm(dim=-1, keepdim=True)).cpu().numpy()
        sh_full = gs["sh0"].data.cpu().numpy()
        N = len(means)
        N_sh = sh_full.shape[1] - 1
        N_rest = N_sh * 3
        ply_path = Path(self.config.result_dir) / f"splats_step{step}.ply"

        with open(ply_path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {N}\n".encode())
            for prop in ["x", "y", "z", "nx", "ny", "nz",
                         "f_dc_0", "f_dc_1", "f_dc_2"]:
                f.write(f"property float {prop}\n".encode())
            for i in range(N_rest):
                f.write(f"property float f_rest_{i}\n".encode())
            f.write(b"property float opacity\n")
            f.write(b"property float scale_0\nproperty float scale_1\n"
                    b"property float scale_2\n")
            f.write(b"property float rot_0\nproperty float rot_1\n"
                    b"property float rot_2\nproperty float rot_3\n")
            f.write(b"end_header\n")

            for i in range(N):
                f.write(struct.pack("<fff", *means[i]))
                f.write(struct.pack("<fff", 1.0, 0.0, 0.0))
                f.write(struct.pack("<fff", *sh_full[i, 0, :]))
                rest = sh_full[i, 1:, :].flatten()
                f.write(struct.pack("<" + "f" * N_rest, *rest))
                f.write(struct.pack("<f", float(opas[i])))
                f.write(struct.pack("<fff", *scales[i]))
                f.write(struct.pack("<ffff", *quats[i]))
        print(f"  PLY: {ply_path} ({N} Gaussians)")

    def _eval(self, gs, transforms, step, sh_degree):
        from PIL import Image
        ensure_local_gsplat()
        from gsplat.rendering import rasterization
        render_dir = Path(self.config.result_dir) / "renders"
        ensure_dir(render_dir)
        cfg = self.config
        with torch.no_grad():
            for i, view in enumerate(transforms[:4]):
                camtoworld = torch.tensor(view["camtoworld"], device=self.device).float()
                K = torch.tensor(view["K"], device=self.device).float()
                W, H = view["width"], view["height"]
                eval_scale = 1.0
                if cfg.data_factor > 1:
                    eval_scale /= cfg.data_factor
                max_dim = max(W, H)
                if max_dim > cfg.max_render_dim:
                    extra = (max_dim + cfg.max_render_dim - 1) // cfg.max_render_dim
                    eval_scale /= extra
                if eval_scale < 1.0:
                    W = max(1, int(W * eval_scale))
                    H = max(1, int(H * eval_scale))
                    K_scaled = K.clone()
                    K_scaled[0, 0] *= eval_scale
                    K_scaled[1, 1] *= eval_scale
                    K_scaled[0, 2] *= eval_scale
                    K_scaled[1, 2] *= eval_scale
                    K = K_scaled
                viewmat = torch.linalg.inv(camtoworld)
                bg = torch.tensor(cfg.background_color, device=self.device).reshape(1, 3)
                render_col, _, _ = rasterization(
                    means=gs["means"],
                    quats=gs["quats"] / gs["quats"].norm(dim=-1, keepdim=True),
                    scales=torch.exp(gs["scales"]),
                    opacities=torch.sigmoid(gs["opacities"]).squeeze(-1),
                    colors=gs["sh0"],
                    viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                    width=W, height=H, sh_degree=sh_degree,
                    camera_model=cfg.camera_model,
                    backgrounds=bg,
                    packed=False,
                )
                rgb = (np.clip(render_col[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                Image.fromarray(rgb).save(render_dir / f"step{step}_view{i}.png")
