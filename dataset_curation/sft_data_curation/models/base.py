from abc import ABC, abstractmethod
from typing import Optional


class BaseGenerator(ABC):
    """
    Base class for all SFT data generation models.
    Subclasses implement load() and generate_response().
    """

    def __init__(self, device: str = "cuda", dtype=None):
        import torch
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.load()

    @abstractmethod
    def load(self):
        """Load model / initialise client."""
        ...

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        video_path: Optional[str] = None,
    ) -> str:
        """
        Generate a response.

        Args:
            prompt:     The full text prompt.
            video_path: Path to the clue video (None for text-only strategies).

        Returns:
            Raw string response from the model.
        """
        ...