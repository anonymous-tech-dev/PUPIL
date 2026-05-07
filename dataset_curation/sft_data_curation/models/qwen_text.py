"""
qwen_text.py
------------
Text-only LLM generator for Qwen3 / Qwen3.5 dense models.
No vision encoder, no video processing — just the LLM backbone.

Use for strategies 1 & 3 (transcript + Q + A, no video frames).
Will raise a clear error if mistakenly used with strategies 2 or 4.
"""

import torch
from typing import Optional

from models.base import BaseGenerator


class QwenTextGenerator(BaseGenerator):
    """
    Qwen3 / Qwen3.5 text-only LLM generator.

    Differences vs QwenVLGenerator:
      - Uses AutoModelForCausalLM (no vision encoder loaded at all)
      - Uses AutoTokenizer instead of AutoProcessor
      - Applies Qwen3's thinking_mode budget token correctly
      - video_path is explicitly unsupported — will raise ValueError
    """

    # ------------------------------------------------------------------ #
    #  Knobs                                                               #
    # ------------------------------------------------------------------ #
    MODEL_ID: str = "Qwen/Qwen3-32B"
    MAX_NEW_TOKENS: int = 2048   # higher default — reasoning traces need room

    # Qwen3 supports a "thinking" budget token.
    # Set to 0 to disable thinking (faster, still high quality for SFT data).
    # Set to e.g. 2048 to enable an internal reasoning scratchpad.
    # Note: the scratchpad is stripped from the final output automatically below.
    THINKING_BUDGET: int = 0

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[QwenText] Loading {self.MODEL_ID} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            dtype=self.dtype,                  # bfloat16 by default
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        print(f"[QwenText] Model ready.")

    # ------------------------------------------------------------------ #
    #  Core generation                                                     #
    # ------------------------------------------------------------------ #
    def generate_response(
        self,
        prompt: str,
        video_path: Optional[str] = None,
    ) -> str:
        if video_path is not None:
            raise ValueError(
                f"QwenTextGenerator does not support video input "
                f"(video_path={video_path!r}). "
                f"Use QwenVLGenerator / Qwen3VLGenerator for strategies 2 & 4."
            )

        messages = [{"role": "user", "content": prompt}]

        # apply_chat_template with thinking budget token (Qwen3-specific).
        # When THINKING_BUDGET=0, thinking is disabled — equivalent to
        # adding /no_think to the system prompt but more explicit.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking=self.THINKING_BUDGET > 0,   # Qwen3 chat template kwarg
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
            )

        # Trim input tokens
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]

        output = self.tokenizer.decode(
            trimmed[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        # Strip the <think>...</think> block if thinking was enabled
        # (we don't want the internal scratchpad in the SFT training target)
        if self.THINKING_BUDGET > 0:
            import re
            output = re.sub(
                r"<think>.*?</think>", "", output,
                flags=re.DOTALL
            ).strip()

        return output


# ---------------------------------------------------------------------------
# Qwen3.5 text subclass — same architecture, just different default MODEL_ID
# ---------------------------------------------------------------------------
class Qwen35TextGenerator(QwenTextGenerator):
    """
    Qwen3.5 text-only LLM (e.g. Qwen3.5-35B-A3B MoE variants).
    Inherits everything from QwenTextGenerator.
    """
    MODEL_ID: str = "Qwen/Qwen3.5-35B-A3B"