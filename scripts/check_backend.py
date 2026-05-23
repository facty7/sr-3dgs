#!/usr/bin/env python3
"""Print optional backend availability."""

from sr_3dgs.backends import backend_status, ensure_local_gsplat

ensure_local_gsplat()

try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as exc:
    print("torch_error", type(exc).__name__, exc)

print(backend_status())

try:
    from gsplat.rendering import rasterization  # noqa: F401
    print("gsplat_rasterization import ok")
except Exception as exc:
    print("gsplat_rasterization_error", type(exc).__name__, exc)
