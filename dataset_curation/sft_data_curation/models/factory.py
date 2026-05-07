"""
Model factory.

Usage
-----
    from models.factory import build_generator
    gen = build_generator("qwen3-32b")
    gen = build_generator("gpt5")
    gen = build_generator("qwen3-32b", model_id_override="/path/to/ckpt")

Strategy compatibility
----------------------
Text-only models (no vision encoder):
    qwen3-32b, qwen3-30b-moe, qwen3.5-35b-moe, qwen3.5-9b
    → strategies 1 & 3 only

Vision-language models:
    qwen3-vl-32b, qwen2.5-7b, qwen2.5-72b
    → all 4 strategies

API models (frames sampled client-side):
    gpt5, gpt4o, gpt-Azure
    → all 4 strategies
"""

from __future__ import annotations
from typing import Optional

from models.base import BaseGenerator


# ---------------------------------------------------------------------------
# Registry: short key -> fully qualified class path
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, str] = {
    # ---- Qwen3-VL vision-language models ----
    "qwen3-vl-8b":   "models.qwen_vl.Qwen3VLGenerator",
    "qwen3-vl-32b":  "models.qwen_vl.Qwen3VLGenerator",
    # ---- Qwen2.5-VL vision-language models ----
    "qwen2.5-7b":    "models.qwen_vl.QwenVLGenerator",
    "qwen2.5-72b":   "models.qwen_vl.QwenVLGenerator",
    # ---- Qwen3 text-only (dense) ----
    "qwen3-32b":     "models.qwen_text.QwenTextGenerator",
    # ---- Qwen3 text-only (MoE) ----
    "qwen3-30b-moe": "models.qwen_text.QwenTextGenerator",
    # ---- Qwen3.5 text (MoE) ----
    "qwen3.5-35b":   "models.qwen_text.Qwen35TextGenerator",
    "qwen3.5-9b":    "models.qwen_text.Qwen35TextGenerator",
    # ---- Azure Azure / GPT ----
    "gpt5":          "models.gpt_Azure.GPTAzureGenerator",
    "gpt4o":         "models.gpt_Azure.GPTAzureGenerator",
    "gpt-Azure":     "models.gpt_Azure.GPTAzureGenerator",
}

# Default HuggingFace MODEL_IDs
_MODEL_ID_MAP: dict[str, str] = {
    "qwen3-vl-8b":   "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-32b":  "Qwen/Qwen3-VL-32B-Instruct",
    "qwen2.5-7b":    "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-72b":   "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen3-32b":     "Qwen/Qwen3-32B",
    "qwen3-30b-moe": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3.5-35b":   "Qwen/Qwen3.5-35B-A3B",
    "qwen3.5-9b":    "Qwen/Qwen3.5-9B",
}

# Default Azure deployment names for Azure keys
_DEPLOYMENT_MAP: dict[str, str] = {
    "gpt5":      "gpt-5.1_2025-11-13",
    "gpt4o":     "gpt-4o_2024-11-20",
    "gpt-Azure": "gpt-5.1_2025-11-13",
}

# Models that CANNOT process video (text-only LLM backbone, no vision encoder)
_TEXT_ONLY_MODELS: set[str] = {
    "qwen3-32b",
    "qwen3-30b-moe",
    "qwen3.5-35b",
    "qwen3.5-9b",
}

# Strategies that require a video-capable model
_VISUAL_STRATEGIES: set[int] = {2, 4}


def validate_model_strategy(model_key: str, strategy: int) -> None:
    """
    Raise a clear error early if a text-only model is paired with a
    visual strategy (2 or 4), rather than crashing mid-run.
    """
    key = model_key.lower().strip()
    if key in _TEXT_ONLY_MODELS and strategy in _VISUAL_STRATEGIES:
        raise ValueError(
            f"Model '{model_key}' is a text-only LLM and cannot be used with "
            f"strategy {strategy} (which requires video frame processing).\n"
            f"  For strategies 2 & 4 use: qwen3-vl-32b, qwen2.5-72b, gpt5, gpt4o\n"
            f"  For strategies 1 & 3 use: qwen3-32b, qwen3-30b-moe, qwen3.5-35b, "
            f"or any vision model"
        )


def build_generator(
    model_key: str,
    model_id_override: Optional[str] = None,
    deployment_override: Optional[str] = None,
    device: str = "cuda",
    strategy: Optional[int] = None,
) -> BaseGenerator:
    """
    Instantiate and return a generator by its short name.

    Parameters
    ----------
    model_key:            Key from the registry (e.g. "qwen3-32b", "gpt5").
    model_id_override:    Override the HuggingFace model ID for local models.
    deployment_override:  Override the Azure deployment name for Azure models.
    device:               Torch device string (ignored for Azure models).
    strategy:             If provided, validates model/strategy compatibility
                          before loading (catches text-model + visual-strategy
                          mismatches early).
    """
    key = model_key.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model key '{model_key}'.\n"
            f"Available: {sorted(_REGISTRY.keys())}"
        )

    # Fail fast on incompatible model/strategy combo
    if strategy is not None:
        validate_model_strategy(key, strategy)

    # Dynamic import
    module_path, class_name = _REGISTRY[key].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Apply defaults then overrides
    effective_model_id = model_id_override or _MODEL_ID_MAP.get(key)
    effective_deployment = deployment_override or _DEPLOYMENT_MAP.get(key)

    # Patch class attributes before instantiation
    if effective_model_id:
        cls = type(cls.__name__, (cls,), {"MODEL_ID": effective_model_id})
    if effective_deployment:
        cls = type(cls.__name__, (cls,), {"DEPLOYMENT_NAME": effective_deployment})

    return cls(device=device)


def list_models() -> list[str]:
    """Print all registered model keys."""
    return sorted(_REGISTRY.keys())


def is_text_only(model_key: str) -> bool:
    return model_key.lower().strip() in _TEXT_ONLY_MODELS