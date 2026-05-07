"""
models/base.py — Abstract base class for VLM backends.
"""

import gc
from abc import ABC, abstractmethod
from typing import List

from PIL import Image


class BaseVLM(ABC):
    """Interface that every VLM backend must implement."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights onto the device."""
        ...

    @abstractmethod
    def call_selection(self, frames: List[Image.Image], prompt: str) -> str:
        """Run a frame-selection call.  Returns raw text."""
        ...

    @abstractmethod
    def call_answering(self, frames: List[Image.Image], prompt: str) -> str:
        """Run an answering call.  Returns raw text."""
        ...

    def unload(self) -> None:
        """Free GPU memory."""
        import torch

        for attr in ("model", "processor"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
