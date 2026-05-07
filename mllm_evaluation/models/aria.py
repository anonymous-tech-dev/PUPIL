"""
Aria-25B (rhymes-ai/Aria) evaluator.

Aria is an MoE multimodal model (25.3B total, 3.9B active) with native
multi-image / long-context support (64K).  It does not have a dedicated
video processor, so per the official cookbook we sample N frames uniformly
and feed them as a list of PIL images, with one `{"type": "image"}` block
per frame in the message.

References:
  * Model card: https://huggingface.co/rhymes-ai/Aria
  * Official multi-image / video cookbook in rhymes-ai/Aria GitHub
"""

import torch
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from transformers import AriaProcessor, AriaForConditionalGeneration

from models.base import BaseEvaluator


class AriaEvaluator(BaseEvaluator):
    MODEL_ID = "rhymes-ai/Aria"
    NUM_FRAMES = 32                # mirror Qwen3-VL default to keep apples-to-apples

    GEN_KWARGS = dict(
        max_new_tokens=1024,
        do_sample=False,
        stop_strings=["<|im_end|>"],
    )

    def load(self):
        self.model = AriaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.processor = AriaProcessor.from_pretrained(self.MODEL_ID)

    # ------------------------------------------------------------------
    def _sample_frames(self, video_path: str):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total = len(vr)
        n = min(self.NUM_FRAMES, total)
        idxs = np.linspace(0, total - 1, num=n, dtype=int)
        arr = vr.get_batch(idxs).asnumpy()              # (T,H,W,3) uint8
        return [Image.fromarray(f) for f in arr]

    # ------------------------------------------------------------------
    def generate_response(self, video_path: str, prompt: str) -> str:
        frames = self._sample_frames(video_path)

        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in frames],
                {"type": "text",
                 "text": (f"{len(frames)} frames are uniformly sampled from a video.\n"
                          f"{prompt}")},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(text=text, images=frames, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        inputs = {k: (v.to(self.model.device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                tokenizer=self.processor.tokenizer,
                **self.GEN_KWARGS,
            )
        out_ids = output[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(out_ids, skip_special_tokens=True).strip()
