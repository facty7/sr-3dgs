#!/usr/bin/env python3
"""Train from an existing aligned/scene_data.npz directory."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.step4_train_3dgs import SR3DGSTrainer, SRTrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--cap_max", type=int, default=120000)
    parser.add_argument("--strategy", choices=("default", "mcmc"), default="default")
    parser.add_argument("--max_render_dim", type=int, default=960)
    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--data_factor", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init_opa", type=float, default=0.5)
    parser.add_argument("--init_scale_multiplier", type=float, default=1.0)
    parser.add_argument("--max_init_scale_fraction", type=float, default=0.10)
    parser.add_argument("--max_train_scale_fraction", type=float, default=0.10)
    parser.add_argument("--refine_start_iter", type=int, default=500)
    parser.add_argument("--refine_stop_iter", type=int, default=12000)
    parser.add_argument("--refine_every", type=int, default=100)
    parser.add_argument("--reset_every", type=int, default=3000)
    parser.add_argument("--grow_grad2d", type=float, default=0.0002)
    parser.add_argument("--prune_opa", type=float, default=0.005)
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--mask_background_weight", type=float, default=0.05)
    parser.add_argument("--mask_alpha_reg", type=float, default=0.05)
    args = parser.parse_args()

    cfg = SRTrainConfig(
        data_dir=args.aligned,
        result_dir=args.out,
        max_steps=args.steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        strategy=args.strategy,
        cap_max=args.cap_max,
        max_render_dim=args.max_render_dim,
        sh_degree=args.sh_degree,
        data_factor=args.data_factor,
        device=args.device,
        init_opa=args.init_opa,
        init_scale_multiplier=args.init_scale_multiplier,
        max_init_scale_fraction=args.max_init_scale_fraction,
        max_train_scale_fraction=args.max_train_scale_fraction,
        refine_start_iter=args.refine_start_iter,
        refine_stop_iter=args.refine_stop_iter,
        refine_every=args.refine_every,
        reset_every=args.reset_every,
        grow_grad2d=args.grow_grad2d,
        prune_opa=args.prune_opa,
        mask_dir=args.mask_dir,
        mask_background_weight=args.mask_background_weight,
        mask_alpha_reg=args.mask_alpha_reg,
    )
    SR3DGSTrainer(cfg).run()


if __name__ == "__main__":
    main()
