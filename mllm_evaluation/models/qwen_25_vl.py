import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.base import BaseEvaluator
import config


class Qwen2_5_VLEvaluator(BaseEvaluator):
    """
    Qwen2.5-VL-7B-Instruct video baseline aligned with VLMEvalKit's
    `Qwen2.5-VL-7B-Instruct-ForVideo` preset.

    References (commit pinned in repo at temp_repo/VLMEvalKit):
      * vlmeval/config.py  → "Qwen2.5-VL-7B-Instruct-ForVideo"
      * vlmeval/vlm/qwen2_vl/model.py  → Qwen2VLChat.__init__ /
        generate_inner_transformers
    """

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    # ── VLMEvalKit "ForVideo" pixel budgets ─────────────────────────────────
    # config.py L1952-L1959
    MIN_PIXELS    = 128 * 28 * 28
    MAX_PIXELS    = 768 * 28 * 28
    TOTAL_PIXELS  = 24576 * 28 * 28
    FPS           = 2  # model.py L217 default

    # ── Qwen2.5-VL official generation hyperparameters ──────────────────────
    # model.py L186-L213 (greedy-ish sampling recipe shipped by VLMEvalKit)
    GEN_KWARGS = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
    )

    def load(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        # AutoProcessor → matches VLMEvalKit (no use_fast kwarg, accepts the
        # new fast image-processor default in transformers >= 4.x).
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            attn_implementation=config.ATTN_IMPL,
            device_map=self.device,
        )
        self.model.eval()

    def generate_response(self, video_path, prompt):
        # Qwen2.5-VL expects file:// URL form (vlmeval's ensure_video_url).
        video_url = (
            video_path
            if video_path.startswith(("http://", "https://", "file://"))
            else f"file://{video_path}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_url,
                        "min_pixels":   self.MIN_PIXELS,
                        "max_pixels":   self.MAX_PIXELS,
                        "total_pixels": self.TOTAL_PIXELS,
                        "fps":          self.FPS,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

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
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.GEN_KWARGS)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
