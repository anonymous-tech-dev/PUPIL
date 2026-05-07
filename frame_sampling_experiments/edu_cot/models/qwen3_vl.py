"""
models/qwen3_vl.py — Qwen3-VL-8B-Instruct backend.

Token budget:
  Qwen3-VL: patch_size=16, merge_size=2 → 32×32 = 1024 pixels/token.
  128 tokens/frame × 1024 = 131,072 max pixels/frame.

All generation is greedy (no sampling).
"""

import logging
from typing import List

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from models.base import BaseVLM

logger = logging.getLogger("educot.qwen3vl")


class Qwen3VLModel(BaseVLM):
    """Qwen3-VL-8B-Instruct backend."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None

        tpf = cfg.model.tokens_per_frame          # default 128
        self._pixels_per_token = 32 * 32           # 1024
        self._max_pixels = tpf * self._pixels_per_token  # 131,072

    def load(self):
        mid = self.cfg.model.model_id
        logger.info("[Qwen3VL] Loading %s …", mid)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            mid,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.cfg.model.attn_impl,
            device_map=self.cfg.model.device,
        )
        self.processor = AutoProcessor.from_pretrained(mid)
        logger.info("[Qwen3VL] Ready.")

    # ─── Message builder ─────────────────────────────────────────────────

    def _build_messages(
        self,
        frames: List[Image.Image],
        prompt: str,
        is_selection: bool = False,
    ) -> list:
        """
        Build Qwen3 chat messages from frames + prompt.

        Selection call  → interleave FrameID labels with image tokens,
                           add system prompt for JSON enforcement.
        Answering call  → block of images then prompt text.
        """
        content = []

        if "<image>" in prompt:
            # ── Interleaved (selection) ───────────────────────────────
            parts = prompt.split("<image>")
            if len(parts) != len(frames) + 1:
                # Mismatch fallback → block layout
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
            # ── Block layout (answering) ─────────────────────────────
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
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

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
