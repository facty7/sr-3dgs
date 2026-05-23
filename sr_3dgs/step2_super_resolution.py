"""Step 2: Apply super-resolution to all input images.

Runs the selected SR model independently on each frame to produce
high-resolution versions. For video-based models (BasicVSR++), processes
frames as a temporal sequence for better consistency.

Input:  Original images from image_dir
Output: Super-resolved images in sr_output_dir (N× original resolution)
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from .sr_models import get_sr_model
from .sr_models.base import BaseSRModel
from .utils import load_image, save_image, ensure_dir


class SuperResolutionProcessor:
    """Apply super-resolution to all input images using the selected model."""

    def __init__(self, image_dir: str, output_dir: str,
                 sr_model_name: str = "real-esrgan",
                 scale: int = 4,
                 device: str = "cuda",
                 model_kwargs: Optional[dict] = None):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.sr_model_name = sr_model_name
        self.scale = scale
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self._model: Optional[BaseSRModel] = None

    def run(self, force: bool = False):
        """Process all images through the SR model."""
        image_paths = self._get_image_paths()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        # Check if already done
        if self._already_done(image_paths) and not force:
            print(f"[Step2] SR images exist in {self.output_dir}, skipping.")
            return self.output_dir

        ensure_dir(self.output_dir)

        try:
            model = self._get_model()
        except Exception as e:
            print(f"[Step2] SR model load failed: {e}")
            print("[Step2] Falling back to original resolution (copy images).")
            self._copy_originals(image_paths)
            return self.output_dir

        is_temporal = model.requires_temporal_input

        if is_temporal:
            self._process_temporal(image_paths, model)
        else:
            self._process_per_image(image_paths, model)

        print(f"[Step2] Super-resolution complete. Output: {self.output_dir}")
        return self.output_dir

    def _process_per_image(self, image_paths: List[Path], model: BaseSRModel):
        """Process images independently (Real-ESRGAN, DAT, SUPIR)."""
        total = len(image_paths)
        for i, img_path in enumerate(image_paths):
            out_path = self.output_dir / img_path.name
            if out_path.exists():
                print(f"  [{i+1}/{total}] Skipping (exists): {img_path.name}")
                continue
            print(f"  [{i+1}/{total}] Processing: {img_path.name}")
            img = load_image(str(img_path))
            sr_img = model.process_image(img)
            save_image(str(out_path), sr_img)

    def _process_temporal(self, image_paths: List[Path], model: BaseSRModel):
        """Process images as a temporal sequence (BasicVSR++)."""
        print(f"[Step2] Loading {len(image_paths)} frames for temporal SR...")
        images = [load_image(str(p)) for p in image_paths]

        print(f"[Step2] Running temporal super-resolution with {model.name}...")
        sr_images = model.process_batch(images)

        for i, img_path in enumerate(image_paths):
            out_path = self.output_dir / img_path.name
            save_image(str(out_path), sr_images[i])

    def _get_model(self) -> BaseSRModel:
        if self._model is None:
            self._model = get_sr_model(
                self.sr_model_name,
                scale=self.scale,
                device=self.device,
                **self.model_kwargs,
            )
            self._model.load()
            print(f"[Step2] Loaded SR model: {self._model.name}")
        return self._model

    def _get_image_paths(self) -> List[Path]:
        paths = sorted([
            p for p in self.image_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ])
        return paths

    def _already_done(self, image_paths: List[Path]) -> bool:
        """Check if all output images already exist."""
        if not self.output_dir.exists():
            return False
        for p in image_paths:
            if not (self.output_dir / p.name).exists():
                return False
        return True

    def _copy_originals(self, image_paths):
        """Copy original images to output when SR is unavailable."""
        import shutil
        for p in image_paths:
            dst = self.output_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
        print(f"[Step2] Copied {len(image_paths)} images at original resolution.")

    def cleanup(self):
        if self._model is not None:
            self._model.unload()
