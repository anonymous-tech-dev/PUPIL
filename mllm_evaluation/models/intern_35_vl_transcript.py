"""
InternVL3.5-8B on TRANSCRIPT only.

Same backbone as `intern_35_vl.py` (`OpenGVLab/InternVL3_5-8B`); we feed the
SRT transcript as the question text and skip the visual encoder entirely by
passing `pixel_values=None, num_patches_list=[]` to `model.chat`.
"""

import torch
from transformers import AutoTokenizer, AutoModel

from models.base import BaseEvaluator
from models.transcript_base import (
    TRANSCRIPT_SYSTEM,
    build_user_prompt,
    load_transcript,
)


class InternVL35TranscriptEvaluator(BaseEvaluator):
    MODEL_ID = "OpenGVLab/InternVL3_5-8B"

    # Same greedy/deterministic preset as the video evaluator (matches
    # VLMEvalKit baseline) so the only delta is the modality.
    GEN_KWARGS = dict(
        do_sample=False,
        max_new_tokens=4096,
        top_p=None,
    )

    NUM_FRAMES = 0

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.device,
        ).eval()
        # Keep a processor handle present for BaseEvaluator.unload() compat.
        self.processor = self.tokenizer

    def generate_response(self, video_path, prompt):
        try:
            transcript = load_transcript(video_path, keep_timestamps=True)
        except FileNotFoundError as e:
            raise RuntimeError(f"transcript_missing: {e}")

        user_text = build_user_prompt(transcript, prompt)
        # InternVL `model.chat` accepts a single user `question` plus an
        # optional `system_message` argument (set on the conv template).
        # Pre-fix the system prompt so behavior is comparable to the other
        # transcript evaluators.
        question = (
            f"{TRANSCRIPT_SYSTEM}\n\n"
            f"{user_text}"
        )

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=None,
                num_patches_list=[],
                question=question,
                generation_config=self.GEN_KWARGS,
                verbose=False,
            )
        return response


class InternVL3TranscriptEvaluator(InternVL35TranscriptEvaluator):
    """Transcript-only variant of the InternVL3-8B baseline."""

    MODEL_ID = "OpenGVLab/InternVL3-8B"
