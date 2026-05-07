import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.base import BaseEvaluator
import config


class Qwen32VLEvaluator(BaseEvaluator):
    MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

    # ── Official Instruct-model generation hyperparameters ──
    GEN_KWARGS = dict(
        max_new_tokens=16384,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
    )

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            attn_implementation=config.ATTN_IMPL,
            device_map="auto",
        )
        self.model.eval()

    def generate_response(self, video_path, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "fps": 2.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        # return_video_metadata=True wraps each video as (tensor, metadata);
        # the HF processor expects plain tensors — unpack them.
        video_metadata = []
        if video_inputs:
            unpacked = []
            for v in video_inputs:
                if isinstance(v, tuple):
                    unpacked.append(v[0])
                    video_metadata.append(v[1])
                else:
                    unpacked.append(v)
            video_inputs = unpacked

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        proc_kwargs = {**video_kwargs}
        if video_metadata:
            proc_kwargs["video_metadata"] = video_metadata

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            do_resize=False,
            padding=True,
            return_tensors="pt",
            **proc_kwargs,
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
        return output_text[0]