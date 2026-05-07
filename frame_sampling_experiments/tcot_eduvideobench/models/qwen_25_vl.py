"""
models/qwen_25_vl.py — Qwen2.5-VL-7B backend for TCoT.

Key difference from the original evaluator:
  TCoT passes *lists of PIL images* (individual frames) rather than a video
  file path.  We format them as a sequence of image messages so the model sees
  each frame tagged with its FrameID, exactly as the paper's prompt specifies.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX 1 — Token budget (latency fix)
  Paper §4.3: "1024 frames / 128K for Qwen2.5"
  → 128K tokens ÷ 1024 frames = 128 tokens per frame.
  Qwen allocates tokens as ceil(H × W / 784) (one token per 28×28 pixels).
  Without a cap, a 1080p frame costs ~2,645 tokens.
  64 frames × 2,645 = 169,280 tokens — already beyond Qwen's 128K context
  limit before the question text is even added, causing silent truncation and
  extreme latency.
  Fix: set max_pixels = 128 × 784 = 100,352 so every frame costs exactly
  128 tokens, keeping each 64-frame selection call to 8,192 vision tokens.

FIX 2 — Frame-ID interleaving (accuracy fix)
  Stage 1 builds the selection prompt with "<image>" placeholders interleaved
  with FrameID labels (Fig. 3 in paper):
      "FrameID 1: <image>\nFrameID 2: <image>\n..."
  The old _build_messages appended all images first, then the full prompt text
  as a single trailing string.  Qwen cannot align a block of N image tokens
  at the top of the context with N "<image>" placeholders buried later in
  the text — the model had no way to know which image = which FrameID.
  Fix: split the prompt on "<image>", then interleave text segments and image
  tokens so each FrameID label sits immediately before its image token in
  the content list.  For the answering call (no "<image>" placeholders),
  the original block layout is correct and preserved.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
from PIL import Image
from typing import List

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from models.base import BaseVLM
import config

import logging
logger = logging.getLogger(__name__)

# Paper §4.3: 128K context / 1024 frames = 128 tokens/frame.
# Qwen token = 28×28 = 784 pixels  →  128 × 784 = 100,352 max pixels/frame.
_TOKENS_PER_FRAME = 128
_PIXELS_PER_TOKEN = 28 * 28          # 784
_MAX_PIXELS       = _TOKENS_PER_FRAME * _PIXELS_PER_TOKEN   # 100,352


class QwenVLModel(BaseVLM):
    """Qwen2.5-VL-7B-Instruct backend."""

    def load(self):
        print(f"[QwenVL] Loading {config.QWEN_MODEL_ID} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.QWEN_MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation=config.ATTN_IMPL,
            device_map=config.QWEN_DEVICE,
        )
        self.processor = AutoProcessor.from_pretrained(
            config.QWEN_MODEL_ID,
            use_fast=False,
            # local_files_only=True,
        )
        print("[QwenVL] Model loaded.")

    # ──────────────────────────────────────────────────────────────────────────
    # Message builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_messages(self, frames: List[Image.Image], prompt: str,
                        is_selection: bool = False) -> list:
        """
        Build the Qwen chat message list from a frame sequence and prompt text.

        Selection call (is_selection=True, prompt contains "<image>" placeholders):
        - Adds a system prompt enforcing JSON-only output (fixes 40%+ parse failure rate)
        - Interleaves FrameID label text and image tokens so the model correctly
            maps each image to its FrameID when outputting JSON frame_ids.

        Answering call (is_selection=False, no "<image>" placeholders):
        - No system prompt needed (free-form answer is fine)
        - Images passed as a block before the prompt text (correct for Qwen answering)
        """
        # ── Build content (image + text interleaving) ──────────────────────────
        content = []

        if "<image>" in prompt:
            # Selection call: interleave FrameID labels with image tokens.
            # prompt looks like:
            #   "FrameID 1: <image>\nFrameID 2: <image>\n...Question: ..."
            # We split on "<image>" to get N+1 text segments, then slot each
            # image between its surrounding text segments so Qwen can map
            # image tokens to their FrameIDs correctly.
            parts = prompt.split("<image>")

            if len(parts) != len(frames) + 1:
                # Safety fallback: count mismatch — use block layout
                for img in frames:
                    content.append({
                        "type"      : "image",
                        "image"     : img,
                        "max_pixels": _MAX_PIXELS,
                    })
                content.append({"type": "text", "text": prompt})
            else:
                for i, img in enumerate(frames):
                    if parts[i]:
                        content.append({"type": "text", "text": parts[i]})
                    content.append({
                        "type"      : "image",
                        "image"     : img,
                        "max_pixels": _MAX_PIXELS,
                    })
                # Trailing text after the last <image>
                if parts[-1]:
                    content.append({"type": "text", "text": parts[-1]})

        else:
            # Answering call: block of images first, then the full prompt text.
            for img in frames:
                content.append({
                    "type"      : "image",
                    "image"     : img,
                    "max_pixels": _MAX_PIXELS,
                })
            content.append({"type": "text", "text": prompt})

        # ── Assemble message list ──────────────────────────────────────────────
        messages = []

        if is_selection:
            # System prompt enforces JSON-only output for the selection call.
            # Without this, Qwen 7B frequently returns plain English instead of
            # JSON, causing the 40%+ parse failure rate seen in analysis.
            messages.append({
                "role": "system",
                "content": (
                    "You are a video analysis assistant. "
                    "You must respond with valid JSON only. "
                    "Do not include any text, explanation, or markdown outside "
                    "the JSON object."
                ),
            })

        messages.append({"role": "user", "content": content})
        return messages

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    def _infer(self, messages: list, max_new_tokens: int) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(config.QWEN_DEVICE)

        logger.info("Total input tokens: %d", inputs["input_ids"].shape[1])

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