"""Step 3: Align camera intrinsics to super-resolved image dimensions.

Uses pycolmap 4.0 API to read COLMAP output and scale intrinsics.
The scaled data is exported for gsplat training.

CRITICAL: Physical FOV is invariant under resolution change.
Only fx, fy, cx, cy scale with resolution; FOV stays the same.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .utils import (
    read_cameras_binary, read_images_binary, read_points3D_binary,
    get_intrinsics, ensure_dir, qvec2rotmat,
)


class IntrinsicAligner:
    """Scale camera intrinsics to match super-resolved image dimensions."""

    def __init__(self, colmap_sparse_dir: str,
                 sr_image_dir: str,
                 output_dir: str,
                 scale_factor: int = 4):
        self.colmap_sparse_dir = Path(colmap_sparse_dir)
        self.sr_image_dir = Path(sr_image_dir)
        self.output_dir = Path(output_dir)
        self.scale_factor = scale_factor

    def run(self, force: bool = False):
        """Scale intrinsics and prepare training data directory."""
        if self._already_done() and not force:
            print(f"[Step3] Aligned data exists at {self.output_dir}, skipping.")
            return self.output_dir

        ensure_dir(self.output_dir)

        # Load COLMAP data using pycolmap 4.0 API (handles all format versions)
        cameras, images, points = self._load_colmap_data()

        print(f"[Step3] Loaded {len(cameras)} cameras, {len(images)} images, "
              f"{len(points)} 3D points from COLMAP.")

        # Copy/link super-resolved images to output
        sr_images_dir = self.output_dir / "images"
        ensure_dir(sr_images_dir)
        image_name_mapping = self._link_sr_images(images, sr_images_dir)

        # Compute image dimensions and per-axis scale factors
        img_h, img_w = self._get_sr_image_dims(sr_images_dir)
        original_w = list(cameras.values())[0]["width"]
        original_h = list(cameras.values())[0]["height"]
        eff_scale_w = img_w / max(original_w, 1)
        eff_scale_h = img_h / max(original_h, 1)

        # Detect gross aspect ratio mismatch (e.g. COLMAP vs SR image orientation)
        orig_aspect = original_w / max(original_h, 1)
        sr_aspect = img_w / max(img_h, 1)
        if abs(orig_aspect - sr_aspect) > 0.05:
            print(f"[Step3] WARNING: Aspect ratio mismatch! "
                  f"Original={original_w}x{original_h} ({orig_aspect:.3f}), "
                  f"SR={img_w}x{img_h} ({sr_aspect:.3f})")

        if abs(eff_scale_w - 1.0) < 0.01 and abs(eff_scale_h - 1.0) < 0.01:
            print(f"[Step3] SR images at original size ({img_w}x{img_h}). K not scaled.")
        else:
            print(f"[Step3] SR effective scale: {eff_scale_w:.2f}x (W) x {eff_scale_h:.2f}x (H)")
            print(f"         Original {original_w}x{original_h} -> {img_w}x{img_h}")

        # Scale camera intrinsics using per-axis effective scale (NOT a single 'max')
        scaled_cameras = self._scale_cameras(
            cameras, eff_scale_w=eff_scale_w, eff_scale_h=eff_scale_h
        )

        # Build per-image data
        image_data = self._build_image_data(
            images, scaled_cameras, image_name_mapping, img_w, img_h
        )

        # ── CRITICAL VALIDATION ──
        self._validate_intrinsics(image_data, img_w, img_h)

        # Extract sparse point cloud for initialization
        points_xyz, points_rgb = self._extract_points(points)

        # Save all data
        self._save_output(image_data, points_xyz, points_rgb,
                          {"w": eff_scale_w, "h": eff_scale_h})

        print(f"[Step3] Intrinsic alignment complete.")
        print(f"  SR image size: {img_w}x{img_h}")
        print(f"  Output: {self.output_dir}")

        return self.output_dir

    def _load_colmap_data(self) -> tuple:
        """Load COLMAP data using pycolmap 4.0 API with binary fallback."""
        sparse_dir = str(self.colmap_sparse_dir)

        # Try pycolmap first
        try:
            import pycolmap
            rec = pycolmap.Reconstruction(sparse_dir)

            cameras = {}
            for cam_id, cam in rec.cameras.items():
                # Determine model name
                model_name = str(cam.model_name) if hasattr(cam, 'model_name') else "SIMPLE_PINHOLE"
                model_id_map = {
                    "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
                    "RADIAL": 3, "OPENCV": 4, "OPENCV_FISHEYE": 5, "FULL_OPENCV": 6,
                }
                model_id = model_id_map.get(model_name, 0)
                params = cam.params.copy() if hasattr(cam.params, 'copy') else np.array(cam.params)

                cameras[cam_id] = {
                    "model_id": model_id,
                    "model_name": model_name,
                    "width": cam.width,
                    "height": cam.height,
                    "params": params,
                }

            images = {}
            for img_id, img in rec.images.items():
                camera_id = img.camera_id if hasattr(img, 'camera_id') else 1
                if hasattr(img, 'cam_from_world'):
                    cfw = img.cam_from_world()
                    world_from_cam = cfw.inverse()
                    camtoworld = np.eye(4)
                    camtoworld[:3, :] = np.array(world_from_cam.matrix())
                    qvec = None
                    tvec = None
                else:
                    qvec = np.array([1.0, 0.0, 0.0, 0.0])
                    tvec = np.array([0.0, 0.0, 0.0])
                    camtoworld = None

                images[img_id] = {
                    "qvec": qvec,
                    "tvec": tvec,
                    "camtoworld": camtoworld,
                    "camera_id": camera_id,
                    "name": img.name,
                }

            points = {}
            for p_id, p in rec.points3D.items():
                points[p_id] = {
                    "xyz": p.xyz.copy() if hasattr(p.xyz, 'copy') else np.array(p.xyz),
                    "rgb": p.color.copy() if hasattr(p, 'color') else np.array(p.color),
                    "error": float(p.error) if hasattr(p, 'error') else 0.0,
                }

            return cameras, images, points

        except Exception as e:
            print(f"[Step3] pycolmap load failed: {e}, trying binary loaders...")

        # Fallback: manual binary reading
        sparse_path = Path(sparse_dir)
        cam_path = sparse_path / "cameras.bin"
        img_path = sparse_path / "images.bin"
        pts_path = sparse_path / "points3D.bin"

        if cam_path.exists() and img_path.exists():
            cameras = read_cameras_binary(str(cam_path))
            images = read_images_binary(str(img_path))
            points = read_points3D_binary(str(pts_path)) if pts_path.exists() else {}
            print(f"[Step3] Loaded via binary readers: {len(cameras)} cameras, "
                  f"{len(images)} images, {len(points)} points.")
            return cameras, images, points

        raise RuntimeError(f"Failed to load COLMAP data from {sparse_dir}")

    def _scale_cameras(self, cameras: Dict,
                       eff_scale_w: float = None,
                       eff_scale_h: float = None) -> Dict:
        """Scale camera intrinsics by per-axis SR scale factor.

        CRITICAL: fx and cx scale with width; fy and cy scale with height.
        FOV is NOT changed by resolution scaling — only the pixel-space
        representation (fx, fy, cx, cy) scales.

        Using a single 'max' factor for both axes is WRONG when the SR
        output has a different aspect ratio than the original COLMAP images.
        """
        if eff_scale_w is None:
            eff_scale_w = float(self.scale_factor)
        if eff_scale_h is None:
            eff_scale_h = float(self.scale_factor)

        scaled = {}
        for cam_id, cam in cameras.items():
            new_cam = dict(cam)
            new_cam["width"] = int(round(cam["width"] * eff_scale_w))
            new_cam["height"] = int(round(cam["height"] * eff_scale_h))

            params = cam["params"].copy()
            model_id = cam["model_id"]
            sw = eff_scale_w
            sh = eff_scale_h

            if model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
                # f is shared, use geometric mean of scale factors
                f_scale = (sw * sh) ** 0.5
                params[0] *= f_scale
                params[1] *= sw  # cx
                params[2] *= sh  # cy
            elif model_id == 1:  # PINHOLE: fx, fy, cx, cy
                params[0] *= sw  # fx
                params[1] *= sh  # fy
                params[2] *= sw  # cx
                params[3] *= sh  # cy
            elif model_id in (2, 3):  # SIMPLE_RADIAL, RADIAL
                f_scale = (sw * sh) ** 0.5
                params[0] *= f_scale  # f
                params[1] *= sw  # cx
                params[2] *= sh  # cy
                # k1, k2 are radial distortion — do NOT scale
            elif model_id in (4, 5, 6):  # OPENCV variants
                params[0] *= sw  # fx
                params[1] *= sh  # fy
                params[2] *= sw  # cx
                params[3] *= sh  # cy
                # k1,k2,p1,p2 are distortion — do NOT scale

            new_cam["params"] = params
            scaled[cam_id] = new_cam

        return scaled

    def _validate_intrinsics(self, image_data: list, img_w: int, img_h: int):
        """Validate that all camera intrinsics are consistent with image dimensions.

        Catches the class of bugs where K matrix values don't match the
        actual image width/height (e.g. when SR scale wasn't applied to K).
        """
        print(f"[Step3] Validating intrinsics for {len(image_data)} images "
              f"(target: {img_w}x{img_h})...")

        for data in image_data:
            K = data["K"]
            W = data["width"]
            H = data["height"]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # Physical FOV from fx and image width should be REASONABLE
            fov_x = 2.0 * np.arctan(W / (2.0 * fx)) * 180.0 / np.pi
            fov_y = 2.0 * np.arctan(H / (2.0 * fy)) * 180.0 / np.pi

            # Assertions that would catch the radial-artifact bug:
            # 1. cx should be near W/2 (principal point near image center)
            assert 0.3 * W <= cx <= 0.7 * W, \
                f"cx={cx:.1f} is too far from W/2={W/2:.1f} (image={W}x{H}). " \
                f"K matrix may not be scaled correctly for SR resolution."

            assert 0.3 * H <= cy <= 0.7 * H, \
                f"cy={cy:.1f} is too far from H/2={H/2:.1f} (image={W}x{H}). " \
                f"K matrix may not be scaled correctly for SR resolution."

            # 2. FOV should be between 30° and 120° (typical cameras)
            assert 8.0 <= fov_x <= 160.0, \
                f"fov_x={fov_x:.1f}° is outside reasonable range [8°, 160°]. " \
                f"fx={fx:.1f}, W={W}. K scaling may be wrong."

            assert 8.0 <= fov_y <= 160.0, \
                f"fov_y={fov_y:.1f}° is outside reasonable range [8°, 160°]. " \
                f"fy={fy:.1f}, H={H}. K scaling may be wrong."

            # 3. fx and fy should be roughly similar (non-anamorphic lens)
            ratio = max(fx, fy) / max(min(fx, fy), 1e-6)
            assert ratio < 2.0, \
                f"fx/fy ratio={ratio:.2f} is suspicious. " \
                f"fx={fx:.1f}, fy={fy:.1f}. Per-axis scaling may differ."

            # 4. Image dims must match K scale
            assert W == img_w, \
                f"Image width mismatch: data says {W}, SR images are {img_w}"
            assert H == img_h, \
                f"Image height mismatch: data says {H}, SR images are {img_h}"

        # Print summary for first camera
        K0 = image_data[0]["K"]
        fov_x0 = 2.0 * np.arctan(image_data[0]["width"] / (2.0 * K0[0, 0])) * 180.0 / np.pi
        fov_y0 = 2.0 * np.arctan(image_data[0]["height"] / (2.0 * K0[1, 1])) * 180.0 / np.pi
        print(f"[Step3]   Sample K[0]: fx={K0[0,0]:.1f}, fy={K0[1,1]:.1f}, "
              f"cx={K0[0,2]:.1f}, cy={K0[1,2]:.1f}")
        print(f"[Step3]   FOV: {fov_x0:.2f}° x {fov_y0:.2f}°")
        print(f"[Step3] All {len(image_data)} cameras passed validation.")

    def _link_sr_images(self, images: Dict, target_dir: Path) -> Dict:
        """Copy or link SR images to output directory."""
        mapping = {}
        for img_id, img in images.items():
            name = img["name"]
            src = self.sr_image_dir / name
            dst = target_dir / name
            if src.exists():
                if not dst.exists():
                    try:
                        os.link(str(src), str(dst))
                    except OSError:
                        shutil.copy2(str(src), str(dst))
                mapping[img_id] = name
            else:
                stem = Path(name).stem
                found = False
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt_src = self.sr_image_dir / (stem + ext)
                    if alt_src.exists():
                        dst_alt = target_dir / (stem + ext)
                        if not dst_alt.exists():
                            try:
                                os.link(str(alt_src), str(dst_alt))
                            except OSError:
                                shutil.copy2(str(alt_src), str(dst_alt))
                        mapping[img_id] = stem + ext
                        found = True
                        break
                if not found:
                    print(f"  WARNING: SR image for {name} not found in {self.sr_image_dir}")
        print(f"[Step3] Linked {len(mapping)} SR images to output.")
        return mapping

    def _get_sr_image_dims(self, sr_images_dir: Path):
        """Get dimensions of super-resolved images."""
        from PIL import Image
        for p in sorted(sr_images_dir.glob("*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                with Image.open(p) as img:
                    return img.height, img.width
        raise RuntimeError(f"No SR images found in {sr_images_dir}")

    def _build_image_data(self, images: Dict, cameras: Dict,
                          name_mapping: Dict, img_w: int, img_h: int) -> list:
        """Build list of per-image data compatible with gsplat format."""
        image_data = []
        for img_id in sorted(images.keys()):
            img = images[img_id]
            if img_id not in name_mapping:
                continue

            cam = cameras[img["camera_id"]]

            # Build camera-to-world matrix
            if img.get("camtoworld") is not None:
                camtoworld = np.array(img["camtoworld"], dtype=np.float64)
            else:
                # COLMAP binary qvec/tvec is world-to-camera; invert it.
                R = qvec2rotmat(img["qvec"])
                t = img["tvec"]
                world_to_cam = np.eye(4)
                world_to_cam[:3, :3] = R
                world_to_cam[:3, 3] = t
                camtoworld = np.linalg.inv(world_to_cam)

            # Build intrinsics matrix K
            fx, fy, cx, cy = get_intrinsics(cam)
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=np.float32)

            image_data.append({
                "image_name": name_mapping[img_id],
                "camtoworld": camtoworld,
                "K": K,
                "width": img_w,
                "height": img_h,
            })

        return image_data

    def _extract_points(self, points: Dict):
        """Extract point cloud from COLMAP points."""
        points_xyz = np.stack([p["xyz"] for p in points.values()], axis=0)
        points_rgb = np.stack([p["rgb"] for p in points.values()], axis=0)
        return points_xyz, points_rgb

    def _save_output(self, image_data: list,
                     points_xyz: np.ndarray, points_rgb: np.ndarray,
                     eff_scale: dict):
        """Save all processed data to output directory."""
        transforms = []
        for d in image_data:
            transforms.append({
                "image_name": d["image_name"],
                "camtoworld": d["camtoworld"].tolist(),
                "K": d["K"].tolist(),
                "width": d["width"],
                "height": d["height"],
            })

        np.savez(
            str(self.output_dir / "scene_data.npz"),
            transforms=transforms,
            points_xyz=points_xyz,
            points_rgb=points_rgb,
            scale_factor_w=eff_scale["w"],
            scale_factor_h=eff_scale["h"],
        )

        metadata = {
            "num_images": len(transforms),
            "num_points": len(points_xyz),
            "scale_factor_w": eff_scale["w"],
            "scale_factor_h": eff_scale["h"],
            "images": transforms,
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _already_done(self) -> bool:
        return (self.output_dir / "scene_data.npz").exists()
