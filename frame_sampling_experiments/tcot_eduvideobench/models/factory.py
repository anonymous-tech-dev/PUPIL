"""
models/factory.py — Instantiate the correct VLM based on config.MODEL.
"""
import config
from models.base import BaseVLM


def get_model() -> BaseVLM:
    name = config.MODEL.lower()
    if "qwen" in name and "3" in name:
        from models.qwen3_vl import Qwen3VLModel
        return Qwen3VLModel()
    elif "qwen" in name:
        from models.qwen_25_vl import QwenVLModel
        return QwenVLModel()
    elif "gpt" in name:
        from models.gpt_Azure import GPTAzureModel
        return GPTAzureModel()
    else:
        raise ValueError(
            f"Unknown model: {config.MODEL!r}. "
            "Choose 'Qwen3-VL-8B', 'Qwen2.5-VL-7B', or a 'GPT-*' deployment."
        )
