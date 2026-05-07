"""
models/qwen_25_vl.py — Qwen2.5-VL-7B-Instruct backend.

Token budget:
  Qwen2.5-VL: patch_size=14, merge_size=2 → 28×28 = 784 pixels/token.
  128 tokens/frame × 784 = 100,352 max pixels/frame.
"""

import logging
from typing import List

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from models.base import BaseVLM

logger = logging.getLogger("educot.qwen25vl")


class Qwen25VLModel(BaseVLM):
    """Qwen2.5-VL-7B-Instruct backend."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None

        tpf = cfg.model.tokens_per_frame
        self._pixels_per_token = 28 * 28           # 784
        self._max_pixels = tpf * self._pixels_per_token  # 100,352

    def load(self):
        mid = self.cfg.model.model_id
        logger.info("[Qwen2.5VL] Loading %s …", mid)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mid,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.cfg.model.attn_impl,
            device_map=self.cfg.model.device,
        )
        self.processor = AutoProcessor.from_pretrained(mid)
        logger.info("[Qwen2.5VL] Ready.")

    def _build_messages(self, frames: List[Image.Image], prompt: str,
                        is_selection: bool = False) -> list:
        content = []

        if "<image>" in prompt:
            parts = prompt.split("<image>")
            if len(parts) != len(frames) + 1:
                for img in frames:
                    content.append({
                        "type": "image", "image": img,
                        "max_pixels": self._max_pixels,
                    })
                content.append({"type": "text", "text": prompt})
            else:
                for i, img in enumerate(frames):
                    if parts[i]:
                        content.append({"type": "text", "text": parts[i]})
                    content.append({
                        "type": "image", "image": img,
                        "max_pixels": self._max_pixels,
                    })
                if parts[-1]:
                    content.append({"type": "text", "text": parts[-1]})
        else:
            for img in frames:
                content.append({
                    "type": "image", "image": img,
                    "max_pixels": self._max_pixels,
                })
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

    def _infer(self, messages: list, max_new_tokens: int) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.cfg.model.device)

        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def call_selection(self, frames: List[Image.Image], prompt: str) -> str:
        msgs = self._build_messages(frames, prompt, is_selection=True)
        return self._infer(msgs, max_new_tokens=self.cfg.selection.max_tokens)

    def call_answering(self, frames: List[Image.Image], prompt: str) -> str:
        msgs = self._build_messages(frames, prompt, is_selection=False)
        return self._infer(msgs, max_new_tokens=self.cfg.generation.answer_max_tokens)
