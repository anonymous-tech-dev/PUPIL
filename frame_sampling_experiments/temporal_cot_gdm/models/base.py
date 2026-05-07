"""
models/base.py — Abstract base for all VLM backends used by TCoT.

Each concrete subclass must implement:
  - load()                                 → load weights / authenticate
  - call_selection(frames, prompt) → str   → run the frame-selection prompt
  - call_answering(frames, prompt) → str   → run the answering prompt
  - unload()                               → free GPU / API resources
"""

import gc
import torch
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseVLM(ABC):
    """Abstract base for Vision-Language Models used in TCoT."""

    def __init__(self):
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self):
        """Load model weights or set up API credentials."""
        raise NotImplementedError

    @abstractmethod
    def call_selection(self, frames: List, prompt: str) -> str:
        """
        Run the VLM frame-selection call (Stage 1 of TCoT).

        Args:
            frames : list of PIL Images or numpy arrays (the segment frames)
            prompt : the formatted selection prompt (Fig. 3 in paper)

        Returns:
            Raw string response from the model (should be JSON).
        """
        raise NotImplementedError

    @abstractmethod
    def call_answering(self, frames: List, prompt: str) -> str:
        """
        Run the VLM answering call (Stage 2 of TCoT).

        Args:
            frames : list of PIL Images or numpy arrays (the curated context)
            prompt : the formatted answering prompt (Fig. 14/15 in paper)

        Returns:
            Raw string response from the model.
        """
        raise NotImplementedError

    def unload(self):
        """Free GPU memory / close connections."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()