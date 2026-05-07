"""
models/qwen3_vl.py — Qwen3-VL backend for TCoT (with optional LoRA adapter).

Same as temporal_cot_gdm/models/qwen3_vl.py but adds:
  • LoRA adapter merging via config.ADAPTER_DIR (env var ADAPTER_DIR).
"""

import logging
import os
import torch
from PIL import Image
from typing import List
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.base import BaseVLM
import config

logger = logging.getLogger(__name__)

# Same token budget as Qwen2.5-VL for direct comparability across experiments.
# Qwen3-VL: patch_size=16, merge_size=2 → 32×32 = 1024 pixels per token.
# 128 tokens/frame × 1024 pixels/token = 131,072 max pixels/frame.
_TOKENS_PER_FRAME = 128
_PIXELS_PER_TOKEN = 32 * 32        # 1024
_MAX_PIXELS       = _TOKENS_PER_FRAME * _PIXELS_PER_TOKEN  # 131,072


class Qwen3VLModel(BaseVLM):
    """Qwen3-VL-8B-Instruct backend with optional LoRA adapter."""

    def load(self):
        model_id = getattr(config, "QWEN_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
        adapter_dir = os.environ.get("ADAPTER_DIR", getattr(config, "ADAPTER_DIR", "")) or ""

        logger.info("[Qwen3VL] Loading %s ...", model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=config.ATTN_IMPL,
            device_map=config.QWEN_DEVICE,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        if adapter_dir and os.path.isdir(adapter_dir):
            from peft import PeftModel
            logger.info("[Qwen3VL] Merging adapter from %s ...", adapter_dir)
            self.model = PeftModel.from_pretrained(self.model, adapter_dir).merge_and_unload()
            logger.info("[Qwen3VL] Adapter merged.")
        else:
            if adapter_dir:
                logger.warning("[Qwen3VL] adapter_dir=%r not found, using base model.", adapter_dir)
            logger.info("[Qwen3VL] Model loaded (base).")

        self.model.eval()

    # ─────────────────────────────────────────────────────────────────────────
    # Message builder — identical logic to qwen_25_vl.py
    # ─────────────────────────────────────────────────────────────────────────
    def _build_messages(self, frames: List[Image.Image], prompt: str,
                        is_selection: bool = False) -> list:
        content = []
        if "<image>" in prompt:
            parts = prompt.split("<image>")
            if len(parts) != len(frames) + 1:
                for img in frames:
                    content.append({"type": "image", "image": img, "max_pixels": _MAX_PIXELS})
                content.append({"type": "text", "text": prompt})
            else:
                for i, img in enumerate(frames):
                    if parts[i]:
                        content.append({"type": "text", "text": parts[i]})
                    content.append({"type": "image", "image": img, "max_pixels": _MAX_PIXELS})
                if parts[-1]:
                    content.append({"type": "text", "text": parts[-1]})
        else:
            for img in frames:
                content.append({"type": "image", "image": img, "max_pixels": _MAX_PIXELS})
            content.append({"type": "text", "text": prompt})

        messages = []
        if is_selection:
            messages.append({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "You are a video analysis assistant. "
                        "You must respond with valid JSON only. "
                        "Do not include any text, explanation, or markdown outside the JSON object."
                    ),
                }],
            })
        messages.append({"role": "user", "content": content})
        return messages

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────
    def _infer(self, messages: list, max_new_tokens: int) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(config.QWEN_DEVICE)

        vision_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_tokens = (inputs["input_ids"] == vision_token_id).sum().item()
        num_frames = len([
            m for m in messages[-1]["content"]
            if isinstance(m, dict) and m.get("type") == "image"
        ])
        logger.info(
            "Total input tokens: %d | Vision tokens: %d | Tokens/frame: %d | Frames: %d",
            inputs["input_ids"].shape[1],
            vision_tokens,
            vision_tokens // num_frames if num_frames > 0 else 0,
            num_frames,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None,
            )
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def call_selection(self, frames: List[Image.Image], prompt: str) -> str:
        msgs = self._build_messages(frames, prompt, is_selection=True)
        return self._infer(msgs, max_new_tokens=config.SELECTION_MAX_TOKENS)

    def call_answering(self, frames: List[Image.Image], prompt: str) -> str:
        msgs = self._build_messages(frames, prompt, is_selection=False)
        return self._infer(msgs, max_new_tokens=config.ANSWER_MAX_TOKENS)

    def unload(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()
