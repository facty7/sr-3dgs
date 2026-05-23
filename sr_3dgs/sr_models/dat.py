"""DAT (Dual Aggregation Transformer) super-resolution model wrapper.

DAT is an alternative "workhorse" model with stronger detail recovery than
Real-ESRGAN on some inputs. Slightly slower but still practical for production.
Best for: images where Real-ESRGAN produces insufficient detail.
"""

import numpy as np
from typing import List

from .base import BaseSRModel
from . import register_model


@register_model("dat")
class DATModel(BaseSRModel):
    """DAT transformer-based SR model."""

    def __init__(self, scale: int = 4, device: str = "cuda", **kwargs):
        super().__init__(scale=scale, device=device, **kwargs)
        self._model = None

    @property
    def name(self) -> str:
        return f"DAT (x{self.scale})"

    def load(self):
        if self._model is not None:
            return
        import torch
        from basicsr.models import create_model as create_basicsr_model

        # DAT uses the BasicSR framework
        # Users should download pretrained weights from:
        # https://github.com/zhengchen1999/DAT
        opt = {
            "name": "DAT",
            "model_type": "SRModel",
            "num_gpu": 1,
            "scale": self.scale,
            "is_train": False,
            "network_g": {
                "type": "DAT",
                "upscale": self.scale,
                "in_chans": 3,
                "img_size": 64,
                "img_range": 1.0,
                "depth": [18],
                "embed_dim": 60,
                "num_heads": [6],
                "split_size": [8, 16],
            },
            "path": {
                "pretrain_network_g": self.kwargs.get("pretrained_path"),
            },
        }
        self._model = create_basicsr_model(opt)
        self._model.to(self.device)
        self._model.eval()

    def process_image(self, img: np.ndarray) -> np.ndarray:
        import torch
        if self._model is None:
            self.load()
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            output = self._model(tensor, return_rgb=True)
        output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return (np.clip(output, 0, 1) * 255).astype(np.uint8)

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [self.process_image(img) for img in images]

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            torch.cuda.empty_cache()
