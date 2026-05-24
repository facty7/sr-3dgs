import os
import json
import struct
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from collections import OrderedDict
from PIL import Image


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    points3D = OrderedDict()
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7], dtype=np.uint8)
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = np.array(track_elems[0::2])
            point2D_idxs = np.array(track_elems[1::2])
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "image_ids": image_ids,
                "point2D_idxs": point2D_idxs,
            }
    return points3D


def read_images_binary(path_to_model_file):
    images = OrderedDict()
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "Iqqqqd")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([
                tuple(map(float, x_y_id_s[0::3])),
                tuple(map(float, x_y_id_s[1::3])),
            ])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
                "xys": xys,
                "point3D_ids": point3D_ids,
            }
    return images


def read_cameras_binary(path_to_model_file):
    cameras = OrderedDict()
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "Iiqq")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = {0: 3, 1: 4, 2: 3, 3: 4, 4: 5, 5: 8, 6: 12}
            params = read_next_bytes(
                fid, 8 * num_params[model_id], "d" * num_params[model_id]
            )
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": np.array(params),
            }
    return cameras


COLMAP_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
}


def get_intrinsics(camera: dict):
    model_id = camera["model_id"]
    params = camera["params"]
    names = {
        0: ("f", "cx", "cy"),
        1: ("fx", "fy", "cx", "cy"),
        2: ("f", "cx", "cy", "k1"),
        3: ("f", "cx", "cy", "k1", "k2"),
        4: ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
        5: ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
        6: ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
    }
    pnames = names.get(model_id, ("fx", "fy", "cx", "cy"))
    if "f" in pnames and "fx" not in pnames:
        return params[0], params[0], params[1], params[2]
    return params[0], params[1], params[2], params[3]


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ],
    ])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def save_image(path, img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_gpu_memory(device: int = 0) -> dict:
    """Get GPU memory info. Returns {'total_gb': float, 'free_gb': float, 'used_gb': float}."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"total_gb": 0, "free_gb": 0, "used_gb": 0}
        free, total = torch.cuda.mem_get_info(device)
        total_gb = total / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        return {"total_gb": total_gb, "free_gb": free_gb, "used_gb": total_gb - free_gb}
    except Exception:
        return {"total_gb": 0, "free_gb": 0, "used_gb": 0}


def vram_safe_config(vram_gb: float, image_w: int, image_h: int) -> dict:
    """Return VRAM-safe training parameters based on available GPU memory.

    Critical values for 3DGS VRAM usage (approximate):
      - Each Gaussian: ~240 bytes (params + optimizer states)
      - 1M Gaussians: ~1.2 GB
      - Rendering at 4K: ~2-4 GB peak
      - Base overhead: ~1 GB

    Returns safe values for: data_factor, sh_degree, max_sh_degree, warn_msg
    """
    H = max(image_w, image_h)
    W = min(image_w, image_h)
    megapixels = (W * H) / 1_000_000

    # Rendering VRAM scales with image area
    render_gb = megapixels * 0.02  # ~20MB per megapixel render buffer

    # How much VRAM is left for Gaussians
    overhead_gb = 1.0
    available_for_gs = vram_gb - overhead_gb - render_gb

    # Each Gaussian with Adam optimizer ≈ 240 bytes (params + grad + m + v)
    max_gs = int(available_for_gs / (240 / 1e9))  # convert to count

    result = {"max_safe_gaussians": max_gs}

    warnings = []

    if vram_gb <= 6:
        result["data_factor"] = max(1, int(H / 1600))
        result["sh_degree"] = 1
        result["sparse_grad"] = True
        warnings.append(f"low VRAM ({vram_gb:.1f} GB): downscaling train resolution, SH degree=1")
    elif vram_gb <= 10:
        if megapixels > 8:  # > 4K
            result["data_factor"] = 2
            warnings.append(">4K images on 8GB: training at 1/2 resolution")
        else:
            result["data_factor"] = 1
        result["sh_degree"] = 3
        result["sparse_grad"] = True
    elif vram_gb <= 16:
        result["data_factor"] = 1
        result["sh_degree"] = 3
        result["sparse_grad"] = True
    else:
        result["data_factor"] = 1
        result["sh_degree"] = 4
        result["sparse_grad"] = False

    result["warnings"] = warnings
    return result


def check_dependencies() -> dict:
    """Check all dependencies and return status dict.

    Call this at pipeline startup to fail fast with clear messages.
    """
    status = {"ok": True, "missing": [], "warnings": []}

    # stdlib — always ok

    # numpy
    try:
        import numpy
        status["numpy"] = numpy.__version__
    except ImportError:
        status["ok"] = False
        status["missing"].append("numpy (pip install numpy)")

    # torch
    try:
        import torch
        status["torch"] = str(torch.__version__)
        if not torch.cuda.is_available():
            status["warnings"].append("CUDA not available — training will be very slow on CPU")
            status["cuda"] = False
        else:
            status["cuda"] = True
            status["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        status["ok"] = False
        status["missing"].append("torch (pip install torch)")

    # PIL
    try:
        from PIL import Image
        status["pillow"] = "OK"
    except ImportError:
        status["ok"] = False
        status["missing"].append("Pillow (pip install Pillow)")

    # cv2
    try:
        import cv2
        status["opencv"] = cv2.__version__
    except ImportError:
        status["ok"] = False
        status["missing"].append("opencv-python (pip install opencv-python)")

    # gsplat
    try:
        from .backends import ensure_local_gsplat
        ensure_local_gsplat()
        import gsplat
        status["gsplat"] = getattr(gsplat, "__version__", "OK")
    except ImportError:
        status["ok"] = False
        status["missing"].append("gsplat (pip install gsplat)")

    # ninja is required by torch/gsplat when CUDA extensions are JIT-compiled.
    import shutil
    ninja_ok = shutil.which("ninja") is not None
    if not ninja_ok:
        try:
            import ninja  # noqa: F401
            ninja_ok = shutil.which("ninja") is not None
        except ImportError:
            pass
    status["ninja"] = "OK" if ninja_ok else "MISSING"
    if not ninja_ok:
        status["ok"] = False
        status["missing"].append("ninja (pip install ninja)")

    # sklearn
    try:
        import sklearn
        status["sklearn"] = sklearn.__version__
    except ImportError:
        status["ok"] = False
        status["missing"].append("scikit-learn (pip install scikit-learn)")

    # torchmetrics
    try:
        import torchmetrics
        status["torchmetrics"] = "OK"
    except ImportError:
        status["warnings"].append("torchmetrics not found — SSIM loss will be skipped")

    # realesrgan (optional)
    try:
        import realesrgan
        status["realesrgan"] = "OK"
    except ImportError:
        status["warnings"].append("realesrgan not found — use --sr_model dat or install realesrgan")

    # imageio
    try:
        import imageio
        status["imageio"] = "OK"
    except ImportError:
        status["warnings"].append("imageio not found — video export disabled")

    # System: colmap (check binary or pycolmap)
    colmap_ok = shutil.which("colmap") is not None
    if not colmap_ok:
        try:
            import pycolmap
            colmap_ok = True
        except ImportError:
            pass
    status["colmap"] = "OK" if colmap_ok else "MISSING"
    if not colmap_ok:
        if not any("colmap" in m for m in status["missing"]):
            status["missing"].append(
                "colmap (pip install pycolmap  or  apt install colmap)"
            )

    # System: ffmpeg (check system + imageio-ffmpeg bundled binary)
    ffmpeg_found = shutil.which("ffmpeg") is not None
    if not ffmpeg_found:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_bin and os.path.exists(ffmpeg_bin):
                ffmpeg_found = True
        except ImportError:
            pass
    status["ffmpeg"] = "OK" if ffmpeg_found else "MISSING"
    if not ffmpeg_found:
        status["warnings"].append("ffmpeg not found — video extraction disabled")

    return status


def print_dep_check(status: dict):
    """Pretty-print dependency check results."""
    print(f"\n{'='*50}")
    print(f"  Dependency Check")
    print(f"{'='*50}")

    core = ["numpy", "torch", "pillow", "opencv", "gsplat", "ninja", "sklearn", "colmap"]
    for k in core:
        v = status.get(k, "MISSING")
        mark = "+" if v and v != "MISSING" else "X"
        vstr = str(v)[:30] if v else "MISSING"
        print(f"  [{mark}] {k:<20} {vstr}")

    opt = ["realesrgan", "torchmetrics", "imageio", "ffmpeg", "cuda"]
    for k in opt:
        v = status.get(k)
        if isinstance(v, bool):
            v = "OK" if v else "NOT FOUND"
        mark = "+" if v and v != "NOT FOUND" else "~"
        vstr = str(v)[:30] if v else "N/A"
        print(f"  [{mark}] {k:<20} {vstr}")

    if status.get("warnings"):
        print(f"\n  Warnings:")
        for w in status["warnings"]:
            print(f"    ! {w}")

    if status.get("missing"):
        print(f"\n  MISSING (required):")
        for m in status["missing"]:
            print(f"    X {m}")

    if not status["ok"]:
        print(f"\n  === 请先安装缺失依赖，或运行: bash scripts/setup_local.sh ===")

    print(f"{'='*50}\n")
