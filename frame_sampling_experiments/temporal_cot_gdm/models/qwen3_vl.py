"""
models/qwen3_vl.py — Qwen3-VL backend for TCoT.

Key differences from qwen_25_vl.py:
  1. Model class: Qwen3VLForConditionalGeneration
     (Qwen2_5_VLForConditionalGeneration in 2.5)
  2. process_vision_info now requires image_patch_size argument:
       process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)
  3. apply_chat_template uses the new tokenize=True, return_dict=True API
     which returns inputs directly without a separate processor() call.
  4. Context window: 256K tokens native (vs 128K for Qwen2.5-VL-7B).
  5. Thinking mode available — we use Instruct (non-thinking) for TCoT
     to match the paper's zero-shot inference setup.

Token budget:
  Qwen3-VL uses the same 32 * 32  pixel = 1 token formula as Qwen2.5-VL.
  _MAX_PIXELS = 128 × 1024 = 131,072 → 128 tokens/frame, matching the
  per-frame budget used throughout our Qwen2.5-VL experiments for
  direct comparability.

All _build_messages logic (system prompt for JSON enforcement, interleaved
FrameID labels for selection, block layout for answering) is identical to
qwen_25_vl.py.  Only the model class and _infer() internals differ.
"""

import logging
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
_PIXELS_PER_TOKEN = 32 * 32        # 1024  (Qwen3-VL patch=16, merge=2)
_MAX_PIXELS       = _TOKENS_PER_FRAME * _PIXELS_PER_TOKEN  # 131,072


class Qwen3VLModel(BaseVLM):
    """Qwen3-VL-8B-Instruct backend."""

    def load(self):
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        logger.info("[Qwen3VL] Loading %s ...", model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,         # must be torch_dtype, not dtype
            attn_implementation=config.ATTN_IMPL,
            device_map=config.QWEN_DEVICE,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        logger.info("[Qwen3VL] Model loaded.")

    # ──────────────────────────────────────────────────────────────────────────
    # Message builder — identical logic to qwen_25_vl.py
    # ──────────────────────────────────────────────────────────────────────────
    def _build_messages(self, frames: List[Image.Image], prompt: str,
                        is_selection: bool = False) -> list:
        """
        Build the Qwen3 chat message list from a frame sequence and prompt.

        Selection call (is_selection=True):
          - System prompt enforces JSON-only output.
          - Interleaves FrameID label text with image tokens so the model
            correctly maps each image to its FrameID in the JSON output.

        Answering call (is_selection=False):
          - No system prompt.
          - Images passed as a block before the prompt text.
        """
        content = []

        if "<image>" in prompt:
            # ── Selection call: interleave labels and images ────────────────
            parts = prompt.split("<image>")
            if len(parts) != len(frames) + 1:
                # Safety fallback: count mismatch → block layout
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
                if parts[-1]:
                    content.append({"type": "text", "text": parts[-1]})

        else:
            # ── Answering call: block of images then prompt text ────────────
            for img in frames:
                content.append({
                    "type"      : "image",
                    "image"     : img,
                    "max_pixels": _MAX_PIXELS,
                })
            content.append({"type": "text", "text": prompt})

        # ── Assemble message list ──────────────────────────────────────────
        messages = []

        if is_selection:
            messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a video analysis assistant. "
                            "You must respond with valid JSON only. "
                            "Do not include any text, explanation, or markdown outside "
                            "the JSON object."
                        ),
                    }
                ],
            })

        messages.append({"role": "user", "content": content})
        return messages

    # ──────────────────────────────────────────────────────────────────────────
    # Inference — uses Qwen3-VL's new apply_chat_template API
    # ──────────────────────────────────────────────────────────────────────────
    def _infer(self, messages: list, max_new_tokens: int) -> str:
        # apply_chat_template(tokenize=True) ignores max_pixels in content dicts.
        # Must use explicit pipeline with process_vision_info + image_patch_size
        # to enforce the per-frame pixel cap.
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

        # vision_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # vision_tokens = (inputs["input_ids"] == vision_token_id).sum().item()
        # logger.info(
        #     "Total input tokens: %d | Vision tokens: %d | Tokens/frame estimate: %d",
        #     inputs["input_ids"].shape[1],
        #     vision_tokens,
        #     vision_tokens // 64 if vision_tokens > 0 else 0,
        # )

        vision_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_tokens = (inputs["input_ids"] == vision_token_id).sum().item()
        num_frames = len([m for m in messages[-1]["content"] if isinstance(m, dict) and m.get("type") == "image"])
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
            # generated_ids = self.model.generate(
            #     **inputs,
            #     max_new_tokens=config.ANSWER_MAX_TOKENS,
            #     do_sample=True,
            #     temperature=0.7,
            #     top_p=0.8,
            #     top_k=20,
            #     repetition_penalty=1.5,
            # )
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