"""
Qwen3-VL-8B-Instruct on TRANSCRIPT only (no video frames).

Loads the same backbone as `qwen_3_vl.py` so the comparison is apples-to-apples:
the only difference between this evaluator and the video-aware Qwen3VLEvaluator
is that the input is the SRT transcript text instead of sampled frames.

This is the open-source half of the "does our benchmark really need the video?"
ablation.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import config
from models.base import BaseEvaluator
from models.transcript_base import (
    TRANSCRIPT_SYSTEM,
    build_user_prompt,
    load_transcript,
)


class Qwen3VLTranscriptEvaluator(BaseEvaluator):
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    # Mirror the canonical sampling preset of the video evaluator so any quality
    # delta is attributable to the modality, not the decoder.
    GEN_KWARGS = dict(
        max_new_tokens=16384,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
    )

    NUM_FRAMES = 0  # captured in the run-metadata sidecar so it's clear

    def load(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            attn_implementation=config.ATTN_IMPL,
            device_map=self.device,
        )
        self.model.eval()

    def _build_inputs(self, video_path: str, prompt: str):
        try:
            transcript = load_transcript(video_path, keep_timestamps=True)
        except FileNotFoundError as e:
            # Surface this as a generation error so the row gets logged with
            # error="..." but the run continues for the next query.
            raise RuntimeError(f"transcript_missing: {e}")

        user_text = build_user_prompt(transcript, prompt)
        messages = [
            {"role": "system", "content": TRANSCRIPT_SYSTEM},
            {"role": "user", "content": user_text},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def generate_response(self, video_path, prompt):
        inputs = self._build_inputs(video_path, prompt)

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
