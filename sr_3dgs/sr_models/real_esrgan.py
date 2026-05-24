"""Real-ESRGAN super-resolution model wrapper.

Real-ESRGAN is the primary "workhorse" model for commercial pipelines.
Fast inference, minimal generative hallucinations, preserves geometric edges.
Best for: images with mild blur, JPEG compression artifacts, or slight noise.
"""

import numpy as np
from pathlib import Path
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
        self._model_scale = 2 if scale == 2 else 4

    @property
    def name(self) -> str:
        return f"Real-ESRGAN ({self.model_type})"

    def describe_weights(self) -> dict:
        """Return weight-resolution metadata without loading the model."""
        explicit = self.kwargs.get("model_path", None)
        explicit_is_url = _is_url(explicit)
        weight_name = self._weight_name()
        release_tag = self._release_tag()
        local_path = Path.home() / ".cache" / "realesrgan" / weight_name
        package_path = Path(__file__).resolve().parents[2] / "weights" / weight_name
        download_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            f"{release_tag}/{weight_name}"
        )

        explicit_exists = None
        explicit_missing = False
        if explicit and not explicit_is_url:
            explicit_path = Path(explicit).expanduser()
            explicit_exists = explicit_path.exists()
            explicit_missing = not explicit_exists

        resolved = self._resolve_model_path()
        needs_download = _is_url(resolved)
        weights_exist = (
            bool(explicit_exists)
            or local_path.exists()
            or package_path.exists()
        )
        return {
            "ok": not explicit_missing,
            "model": self.name,
            "requested_scale": int(self.scale),
            "model_scale": int(self._model_scale),
            "weight_name": weight_name,
            "explicit_model_path": explicit or "",
            "explicit_is_url": bool(explicit_is_url),
            "explicit_exists": explicit_exists,
            "local_cache_path": str(local_path),
            "local_cache_exists": local_path.exists(),
            "package_path": str(package_path),
            "package_exists": package_path.exists(),
            "download_url": download_url,
            "resolved_model_path": resolved,
            "needs_download": bool(needs_download),
            "weights_exist": bool(weights_exist),
        }

    def _weight_name(self) -> str:
        return "RealESRGAN_x2plus.pth" if self._model_scale == 2 else "RealESRGAN_x4plus.pth"

    def _release_tag(self) -> str:
        return "v0.2.1" if self._model_scale == 2 else "v0.1.0"

    def _resolve_model_path(self) -> str:
        model_path = self.kwargs.get("model_path", None)
        if model_path:
            return str(model_path)
        weight_name = self._weight_name()
        local_path = Path.home() / ".cache" / "realesrgan" / weight_name
        package_path = Path(__file__).resolve().parents[2] / "weights" / weight_name
        if local_path.exists():
            return str(local_path)
        if package_path.exists():
            return str(package_path)
        return (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            f"{self._release_tag()}/{weight_name}"
        )

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
                num_block=6, num_grow_ch=32, scale=self._model_scale
            )
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=self._model_scale
            )

        model_url = self._resolve_model_path()
        self._model = RealESRGANer(
            scale=self._model_scale,
            model_path=model_url,
            model=model,
            tile=int(self.kwargs.get("tile", 0)),
            tile_pad=int(self.kwargs.get("tile_pad", 10)),
            pre_pad=int(self.kwargs.get("pre_pad", 0)),
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


def _is_url(value) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))
