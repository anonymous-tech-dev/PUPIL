"""
Qwen2.5-VL-7B-Instruct on TRANSCRIPT only.

Mirrors the video evaluator in `qwen_25_vl.py` (same backbone, same generation
preset) but feeds the SRT transcript as text instead of sampled frames.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import config
from models.base import BaseEvaluator
from models.transcript_base import (
    TRANSCRIPT_SYSTEM,
    build_user_prompt,
    load_transcript,
)


class Qwen2_5_VLTranscriptEvaluator(BaseEvaluator):
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Match the official VLMEvalKit "ForVideo" preset of `qwen_25_vl.py` so
    # the only delta vs. the video baseline is the modality.
    GEN_KWARGS = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
    )

    NUM_FRAMES = 0

    def load(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            attn_implementation=config.ATTN_IMPL,
            device_map=self.device,
        )
        self.model.eval()

    def generate_response(self, video_path, prompt):
        try:
            transcript = load_transcript(video_path, keep_timestamps=True)
        except FileNotFoundError as e:
            raise RuntimeError(f"transcript_missing: {e}")

        user_text = build_user_prompt(transcript, prompt)
        messages = [
            {"role": "system", "content": TRANSCRIPT_SYSTEM},
            {"role": "user", "content": user_text},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.GEN_KWARGS)

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return out[0]
