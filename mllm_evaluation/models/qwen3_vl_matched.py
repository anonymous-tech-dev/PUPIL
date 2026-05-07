"""
Qwen3-VL-8B-Instruct (NO ADAPTER) evaluator — but with the SAME video
preprocessing knobs the SoF DPO model was fine-tuned with.

Use this as the apples-to-apples baseline next to qwen3_vl_ft. The original
qwen_3_vl.py is preserved untouched (it uses qwen-vl-utils defaults: fps=2,
~768 frames, ~19M total px) and represents the "best baseline you can give
Qwen3-VL". This wrapper instead represents "what the baseline scores at the
same low-frame distribution your fine-tune was trained on" — the proper
controlled comparison for measuring lift from the adapter.

Same video / generation env vars as qwen3_vl_finetuned.py:
  VIDEO_FPS, VIDEO_MAX_FRAMES, VIDEO_MIN_FRAMES,
  VIDEO_MAX_PIXELS, VIDEO_MIN_PIXELS, VIDEO_TOTAL_PIXELS,
  GEN_MAX_NEW_TOKENS, GEN_DO_SAMPLE, GEN_TEMPERATURE, GEN_TOP_P, GEN_TOP_K
"""

import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.base import BaseEvaluator
import config


def _envf(name, default):
    return float(os.environ.get(name, str(default)))


def _envi(name, default):
    return int(os.environ.get(name, str(default)))


def _envb(name, default):
    return os.environ.get(name, str(int(default))).strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


class Qwen3VLMatchedEvaluator(BaseEvaluator):
    # Class default — kept for backward compatibility with anything that
    # introspects the class attribute. The instance attribute set in load()
    # respects MODEL_ID env var so we can swap to 32B without code changes.
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def load(self):
        # Allow env override (default = the 8B path that produced the leaderboard).
        self.MODEL_ID = os.environ.get("MODEL_ID", type(self).MODEL_ID)

        # ─── Video preprocessing knobs (match training defaults) ────────────
        self.video_fps          = _envf("VIDEO_FPS",          1.0)
        self.video_max_frames   = _envi("VIDEO_MAX_FRAMES",   8)
        self.video_min_frames   = _envi("VIDEO_MIN_FRAMES",   4)
        self.video_max_pixels   = _envi("VIDEO_MAX_PIXELS",   768 * 32 * 32)
        self.video_min_pixels   = _envi("VIDEO_MIN_PIXELS",   128 * 32 * 32)
        self.video_total_pixels = _envi("VIDEO_TOTAL_PIXELS",
                                        self.video_max_frames * self.video_max_pixels)

        # ─── Generation knobs ───────────────────────────────────────────────
        self.gen_max_new_tokens = _envi("GEN_MAX_NEW_TOKENS", 512)
        self.gen_do_sample      = _envb("GEN_DO_SAMPLE",      False)
        self.gen_temperature    = _envf("GEN_TEMPERATURE",    0.7)
        self.gen_top_p          = _envf("GEN_TOP_P",          0.8)
        self.gen_top_k          = _envi("GEN_TOP_K",          20)

        print(f"📦 Loading base model: {self.MODEL_ID}  (matched-distribution baseline)")
        print( "🎞  Video preprocessing (matched to training):")
        print(f"     fps={self.video_fps}  max_frames={self.video_max_frames}  "
              f"min_frames={self.video_min_frames}")
        print(f"     max_pixels={self.video_max_pixels}  "
              f"min_pixels={self.video_min_pixels}  "
              f"total_pixels={self.video_total_pixels}")
        print( "✏️  Generation:")
        print(f"     max_new_tokens={self.gen_max_new_tokens}  "
              f"do_sample={self.gen_do_sample}  "
              f"T={self.gen_temperature}  top_p={self.gen_top_p}  "
              f"top_k={self.gen_top_k}")

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=self.dtype,
            attn_implementation=config.ATTN_IMPL,
            device_map=self.device,
        )
        self.model.eval()
        print("✅ Qwen3-VL-8B baseline ready.")

    def _build_messages(self, video_path, prompt):
        # See qwen3_vl_finetuned.py for rationale: the SFT/DPO data
        # is wrapped "<video>\nQuestion: {q}". The chat template inserts
        # the vision tokens, so prepending "Question: " here makes the
        # post-template string match training exactly. The matched
        # baseline is meant to be a strict-control comparison against the
        # FT models, so it must use the IDENTICAL prompt skeleton.
        q_text = prompt if str(prompt).lstrip().lower().startswith("question:") \
                        else f"Question: {prompt}"
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}"
                                 if not str(video_path).startswith("file://")
                                 else video_path,
                        "fps":          self.video_fps,
                        "max_frames":   self.video_max_frames,
                        "min_frames":   self.video_min_frames,
                        "max_pixels":   self.video_max_pixels,
                        "min_pixels":   self.video_min_pixels,
                        "total_pixels": self.video_total_pixels,
                    },
                    {"type": "text", "text": q_text},
                ],
            }
        ]

    def generate_response(self, video_path, prompt):
        messages = self._build_messages(video_path, prompt)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

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

        gen_kwargs = dict(max_new_tokens=self.gen_max_new_tokens)
        if self.gen_do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.gen_temperature,
                top_p=self.gen_top_p,
                top_k=self.gen_top_k,
            )
        else:
            gen_kwargs.update(do_sample=False, temperature=None,
                              top_p=None, top_k=None)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

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
