"""BasicVSR++ video super-resolution model wrapper.

BasicVSR++ is the optimal choice for video input because it leverages
temporal information across adjacent frames, naturally producing
multi-view consistent super-resolution.

Best for: continuous video frame input where temporal coherence matters.

NOTE: This model processes video clips (multiple frames at once), not
individual images. It's the only model in the pipeline where
requires_temporal_input = True.
"""

import numpy as np
from typing import List

from .base import BaseSRModel
from . import register_model


@register_model("basicvsr++")
@register_model("basicvsr_plus")
@register_model("basicvsr")
class BasicVSRPlusModel(BaseSRModel):
    """BasicVSR++ video super-resolution model."""

    def __init__(self, scale: int = 4, device: str = "cuda",
                 num_frames: int = 30, **kwargs):
        super().__init__(scale=scale, device=device, **kwargs)
        self.num_frames = num_frames
        self._model = None

    @property
    def name(self) -> str:
        return f"BasicVSR++ (x{self.scale}, {self.num_frames}f)"

    @property
    def requires_temporal_input(self) -> bool:
        return True

    def load(self):
        if self._model is not None:
            return
        import torch

        try:
            from mmedit.apis import init_model, restoration_video_inference
            self._mmedit_inference = restoration_video_inference
        except ImportError:
            try:
                from mmagic.apis import init_model, restoration_video_inference
                self._mmedit_inference = restoration_video_inference
            except ImportError:
                raise ImportError(
                    "BasicVSR++ requires mmediting/mmagic. Install with:\n"
                    "pip install mmagic openmim\n"
                    "mim install mmcv-full\n"
                    "Or use a different SR model: --sr_model real-esrgan"
                )

        # Users should download pretrained weights from:
        # https://github.com/open-mmlab/mmediting
        config = self.kwargs.get("config_path")
        checkpoint = self.kwargs.get("checkpoint_path")
        if not config or not checkpoint:
            raise ValueError(
                "BasicVSR++ requires 'config_path' and 'checkpoint_path' kwargs"
            )

        self._model = init_model(config, checkpoint, device=self.device)

    def process_image(self, img: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "BasicVSR++ requires temporal input. Use process_video() instead."
        )

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Process video frames with temporal propagation."""
        import torch
        if self._model is None:
            self.load()

        # Stack frames with temporal padding
        frames = []
        for img in images:
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            frames.append(img)

        # Pad to multiple of num_frames
        padded = list(frames)
        while len(padded) < self.num_frames:
            padded.append(frames[-1])

        # Process in sliding windows
        results = []
        for i in range(0, len(frames), self.num_frames):
            chunk = padded[i:i + self.num_frames]
            if len(chunk) < self.num_frames:
                chunk = chunk + [chunk[-1]] * (self.num_frames - len(chunk))
            chunk_tensor = torch.from_numpy(
                np.stack(chunk, axis=0).transpose(0, 3, 1, 2)
            ).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self._mmedit_inference(self._model, chunk_tensor)
            # output shape: [1, T, 3, H, W] or list
            output_np = output.squeeze(0).cpu().numpy().transpose(0, 2, 3, 1)
            for j in range(min(len(chunk), len(frames) - i)):
                out_frame = output_np[j]
                results.append(
                    (np.clip(out_frame, 0, 1) * 255).astype(np.uint8)
                )

        return results

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            torch.cuda.empty_cache()
