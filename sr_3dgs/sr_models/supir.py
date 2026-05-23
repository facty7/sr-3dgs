"""SUPIR super-resolution model wrapper.

SUPIR is a diffusion-based "rescue" model for extremely low-quality inputs.
It can generate photorealistic textures from severely degraded images,
but MUST be used with the decoupled pipeline to avoid multi-view inconsistency.

Best for: extremely blurry or degraded source images where GAN-based
models fail completely.

IMPORTANT: SUPIR generates hallucinated textures. Always run COLMAP on
original low-res images BEFORE applying SUPIR, and never the reverse.
"""

import numpy as np
from typing import List, Optional

from .base import BaseSRModel
from . import register_model


@register_model("supir")
class SUPIRModel(BaseSRModel):
    """SUPIR diffusion-based SR model for extreme-quality recovery."""

    def __init__(self, scale: int = 4, device: str = "cuda",
                 prompt: Optional[str] = None,
                 negative_prompt: str = "blur, noise, distortion, low quality",
                 guidance_scale: float = 7.5,
                 num_steps: int = 50,
                 **kwargs):
        super().__init__(scale=scale, device=device, **kwargs)
        self.prompt = prompt or "high quality, sharp, detailed, photorealistic"
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.num_steps = num_steps
        self._pipe = None

    @property
    def name(self) -> str:
        return f"SUPIR (x{self.scale})"

    def load(self):
        if self._pipe is not None:
            return
        import torch

        # SUPIR requires significant setup. Users must install:
        # pip install SUPIR (or clone from https://github.com/Fanghua-Yu/SUPIR)
        try:
            from SUPIR.util import create_SUPIR_model
            from SUPIR.utils import PIL2Tensor, Tensor2PIL
            from CKPT_PTH import SUPIR_cache
        except ImportError:
            raise ImportError(
                "SUPIR is not installed. Install from: "
                "https://github.com/Fanghua-Yu/SUPIR\n"
                "Or use a different SR model: --sr_model real-esrgan"
            )

        self._pipe = create_SUPIR_model(
            supir_sign="Q",
            ckpt=SUPIR_cache,
            color_fix_type="AdaIn",
        )
        self._pipe.to(self.device)

    def process_image(self, img: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        if self._pipe is None:
            self.load()

        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)

        with torch.no_grad():
            result = self._pipe(
                pil_img,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_steps,
            )
        return np.array(result)

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [self.process_image(img) for img in images]

    def unload(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            import torch
            torch.cuda.empty_cache()
