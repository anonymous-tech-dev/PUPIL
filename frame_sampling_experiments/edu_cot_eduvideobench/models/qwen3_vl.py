"""
models/qwen3_vl.py — Qwen3-VL-8B-Instruct backend (Pupil edition).

Changes from edu_cot/models/qwen3_vl.py:
  • Optional LoRA adapter merging — set cfg.model.adapter_dir or env ADAPTER_DIR.
"""

import logging
import os
from typing import List

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from models.base import BaseVLM

logger = logging.getLogger("educot.qwen3vl")


class Qwen3VLModel(BaseVLM):
    """Qwen3-VL-8B-Instruct backend with optional LoRA adapter."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None

        tpf = cfg.model.tokens_per_frame                   # default 128
        self._pixels_per_token = 32 * 32                    # 1024
        self._max_pixels = tpf * self._pixels_per_token     # 131,072 by default

    def load(self):
        mid = self.cfg.model.model_id
        adapter_dir = (
            (self.cfg.model.get("adapter_dir") or "")
            or os.environ.get("ADAPTER_DIR", "")
        )

        logger.info("[Qwen3VL] Loading %s …", mid)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            mid,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.cfg.model.attn_impl,
            device_map=self.cfg.model.device,
        )
        self.processor = AutoProcessor.from_pretrained(mid)

        if adapter_dir and os.path.isdir(adapter_dir):
            from peft import PeftModel
            logger.info("[Qwen3VL] Loading LoRA adapter from %s …", adapter_dir)
            self.model = PeftModel.from_pretrained(self.model, adapter_dir).merge_and_unload()
            logger.info("[Qwen3VL] Adapter merged. Ready.")
        else:
            if adapter_dir:
                logger.warning("[Qwen3VL] adapter_dir=%r not found, using base model.", adapter_dir)
            logger.info("[Qwen3VL] Ready (base model).")

        self.model.eval()

    # ─── Message builder ─────────────────────────────────────────────────
    def _build_messages(self, frames: List[Image.Image], prompt: str,
                        is_selection: bool = False) -> list:
        content = []
        if "<image>" in prompt:
            parts = prompt.split("<image>")
            if len(parts) != len(frames) + 1:
                for img in frames:
                    content.append({"type": "image", "image": img,
                                    "max_pixels": self._max_pixels})
                content.append({"type": "text", "text": prompt})
            else:
                for i, img in enumerate(frames):
                    if parts[i]:
                        content.append({"type": "text", "text": parts[i]})
                    content.append({"type": "image", "image": img,
                                    "max_pixels": self._max_pixels})
                if parts[-1]:
                    content.append({"type": "text", "text": parts[-1]})
        else:
            for img in frames:
                content.append({"type": "image", "image": img,
                                "max_pixels": self._max_pixels})
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
                        "Do not include any text, explanation, or markdown "
                        "outside the JSON object."
                    ),
                }],
            })
        messages.append({"role": "user", "content": content})
        return messages

    # ─── Inference ────────────────────────────────────────────────────────
    def _infer(self, messages: list, max_new_tokens: int) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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
        ).to(self.cfg.model.device)

        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # ─── Public API ───────────────────────────────────────────────────────
    def call_selection(self, frames: List[Image.Image], prompt: str) -> str:
        msgs = self._build_messages(frames, prompt, is_selection=True)
        return self._infer(msgs, max_new_tokens=self.cfg.selection.max_tokens)

    def call_answering(self, frames: List[Image.Image], prompt: str) -> str:
        msgs = self._build_messages(frames, prompt, is_selection=False)
        return self._infer(msgs, max_new_tokens=self.cfg.generation.answer_max_tokens)

    def unload(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()
