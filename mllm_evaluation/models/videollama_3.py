"""
VideoLLaMA3-7B evaluator.

Reference: official HF model card quick-start
  https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B

Note: VLMEvalKit does NOT include VideoLLaMA3, so we ground directly on the
HF readme's recommended defaults:
  * `AutoModelForCausalLM` + `AutoProcessor` with `trust_remote_code=True`
  * Conversation format with `{"type": "video", "video": {"video_path", "fps", "max_frames"}}`
  * `fps=1, max_frames=128` (the readme example values)
  * `attn_implementation="flash_attention_2"`
  * `pixel_values` cast to bfloat16 after processor call

The HF quick-start uses `max_new_tokens=128` for a single-line description; we
use 1024 to give room for the longer, paragraph-style answers our benchmark
elicits.

The model card's text-chat example mentions sampling defaults that ship in the
shipped `generation_config.json` for VideoLLaMA3-7B (do_sample=True). We leave
`do_sample` unspecified so HF inherits whatever the model author shipped — this
mirrors VLMEvalKit's strategy for Qwen3-VL and avoids overriding the author's
recommended decoder.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from models.base import BaseEvaluator
import config


class VideoLLaMA3Evaluator(BaseEvaluator):
    MODEL_ID = "DAMO-NLP-SG/VideoLLaMA3-7B"

    # Frame-sampling matches the official quick-start
    NUM_FRAMES = 128   # max_frames in the video processor (sidecar metadata)
    FPS = 1

    GEN_KWARGS = dict(
        max_new_tokens=1024,
    )

    SYSTEM_PROMPT = "You are a helpful assistant."

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            attn_implementation=config.ATTN_IMPL,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )

    def generate_response(self, video_path, prompt):
        conversation = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": self.FPS,
                            "max_frames": self.NUM_FRAMES,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor(conversation=conversation, return_tensors="pt")
        inputs = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self.GEN_KWARGS)

        # Trim the prompt portion before decode so we only return the assistant turn.
        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[:, input_len:]
        response = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True
        )[0].strip()
        return response
