"""Real-ESRGAN super-resolution model wrapper.

Real-ESRGAN is the primary "workhorse" model for commercial pipelines.
Fast inference, minimal generative hallucinations, preserves geometric edges.
Best for: images with mild blur, JPEG compression artifacts, or slight noise.
"""

import numpy as np
from typing import List

from .base import BaseSRModel
from . import register_model


@register_model("real-esrgan")
@register_model("realesrgan")
class RealESRGANModel(BaseSRModel):
    """Real-ESRGAN wrapper supporting both general and anime variants."""

    def __init__(self, scale: int = 4, device: str = "cuda",
                 model_type: str = "realesr-general-x4v3", **kwargs):
        super().__init__(scale=scale, device=device, **kwargs)
        self.model_type = model_type
        self._model = None

    @property
    def name(self) -> str:
        return f"Real-ESRGAN ({self.model_type})"

    def load(self):
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if self._model is not None:
            return

        # Determine model architecture based on model_type
        if "anime" in self.model_type:
            # Anime models use a different RRDBNet config
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=self.scale
            )
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=self.scale
            )

        # Use standard model URL, try multiple sources
        model_url = self.kwargs.get("model_path", None)
        if model_url is None:
            # Try local weights directory first
            import os as _os
            local_path = _os.path.expanduser("~/.cache/realesrgan/RealESRGAN_x4plus.pth")
            package_path = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                "weights",
                "RealESRGAN_x4plus.pth",
            )
            if _os.path.exists(local_path):
                model_url = local_path
            elif _os.path.exists(package_path):
                model_url = package_path
            else:
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        self._model = RealESRGANer(
            scale=self.scale,
            model_path=model_url,
            model=model,
            tile=0,  # No tiling by default; set tile>0 if OOM
            tile_pad=10,
            pre_pad=0,
            half=True if self.device == "cuda" else False,
            device=self.device,
        )

    def process_image(self, img: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()
        if img.dtype == np.float32 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        output, _ = self._model.enhance(img, outscale=self.scale)
        return output

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [self.process_image(img) for img in images]

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            torch.cuda.empty_cache()
