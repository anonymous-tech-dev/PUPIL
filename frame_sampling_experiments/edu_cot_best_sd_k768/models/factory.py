"""
models/factory.py — Instantiate the correct VLM backend from config.
"""

from omegaconf import DictConfig
from models.base import BaseVLM


def get_model(cfg: DictConfig) -> BaseVLM:
    name = cfg.model.name.lower()

    if "qwen" in name and "3" in name:
        from models.qwen3_vl import Qwen3VLModel
        return Qwen3VLModel(cfg)
    elif "qwen" in name:
        from models.qwen_25_vl import Qwen25VLModel
        return Qwen25VLModel(cfg)
    else:
        raise ValueError(
            f"Unknown model: {cfg.model.name!r}. "
            "Supported: 'Qwen3-VL-8B', 'Qwen2.5-VL-7B'."
        )
