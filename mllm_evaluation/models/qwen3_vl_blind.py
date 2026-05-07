"""
Qwen3-VL-8B-Instruct in *blind* mode (no video frames, no transcript).

This is the strictest open-source baseline: same backbone and generation
preset as `qwen_3_vl.py` (and `qwen3_vl_transcript.py`), but the input is
ONLY the question text — no video and no SRT. Used to quantify how much of
Pupil can be solved from the model's pre-training knowledge alone,
without any conditioning on the lecture.

Any quality gap between this and the video / transcript variants is therefore
attributable purely to the modality, not to the decoder or the prompt format.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import config
from models.base import BaseEvaluator
from models.blind_base import BLIND_SYSTEM, build_blind_prompt


class Qwen3VLBlindEvaluator(BaseEvaluator):
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    # Mirror the canonical sampling preset of the video / transcript evaluators
    # so any quality delta is attributable to the modality, not the decoder.
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

        # We still load the multimodal AutoProcessor (cheap), but only use its
        # tokenizer.  Keeping it identical to the video / transcript evaluators
        # ensures the chat template + special tokens are byte-for-byte the same.
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            attn_implementation=config.ATTN_IMPL,
            device_map=self.device,
        )
        self.model.eval()

    def _build_inputs(self, prompt: str):
        user_text = build_blind_prompt(prompt)
        messages = [
            {"role": "system", "content": BLIND_SYSTEM},
            {"role": "user", "content": user_text},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def generate_response(self, video_path, prompt):
        # video_path is intentionally unused — this is the *blind* baseline.
        del video_path
        inputs = self._build_inputs(prompt)

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
