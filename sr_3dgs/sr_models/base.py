"""Base class for super-resolution models."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseSRModel(ABC):
    """Abstract interface for all SR models in the pipeline.

    Each subclass handles one SR backend (Real-ESRGAN, DAT, SUPIR, BasicVSR++).
    """

    def __init__(self, scale: int = 4, device: str = "cuda", **kwargs):
        self.scale = scale
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def load(self):
        """Load the model weights into memory."""

    @abstractmethod
    def process_image(self, img: np.ndarray) -> np.ndarray:
        """Super-resolve a single image.

        Args:
            img: RGB image as uint8 [H, W, 3] or float32 [H, W, 3] in [0, 1].

        Returns:
            Super-resolved RGB image as uint8 [H*scale, W*scale, 3].
        """

    @abstractmethod
    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Super-resolve a batch of images (default: loop over process_image)."""

    @abstractmethod
    def unload(self):
        """Release GPU memory."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @property
    def requires_temporal_input(self) -> bool:
        """Whether this model requires temporal (video frame) input."""
        return False
