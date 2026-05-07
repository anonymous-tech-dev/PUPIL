"""
Qwen3-VL-8B + LoRA adapter evaluator for Pupil.

Loads base Qwen3-VL-8B-Instruct, merges a LoRA adapter from ADAPTER_DIR,
then runs the native-video pipeline with VIDEO PRE-PROCESSING PARAMETERS
EXACTLY MATCHING TRAINING (so the frozen LM sees the same input distribution
it was fine-tuned on).

────────────────────────────────────────────────────────────────────────────
Adapter env vars
  ADAPTER_DIR  — path to the LoRA adapter directory (required)
  ADAPTER_TAG  — optional short name for output folders
                 (default: basename of ADAPTER_DIR)

Video preprocessing env vars (defaults mirror the SoF DPO training script
contrastive_experiments/sof_dpo/scripts/20_sof_dpo.sh):
  VIDEO_FPS            (default 1.0)         — frames per second to sample
  VIDEO_MAX_FRAMES     (default 8)           — hard cap on frames per video
  VIDEO_MIN_FRAMES     (default 4)           — floor (qwen-vl-utils default)
  VIDEO_MAX_PIXELS     (default 786432 = 768*32*32) — max px / frame
  VIDEO_MIN_PIXELS     (default 131072 = 128*32*32) — min px / frame
  VIDEO_TOTAL_PIXELS   (default 6291456 = 8*768*32*32) — total px / video

Generation env vars (default = greedy because DPO models are typically
better-decoded greedily; flip via env if you need to match baseline sampling):
  GEN_MAX_NEW_TOKENS   (default 512)
  GEN_DO_SAMPLE        (default 0 = greedy; set 1 to enable sampling)
  GEN_TEMPERATURE      (default 0.7, only used if sampling)
  GEN_TOP_P            (default 0.8, only used if sampling)
  GEN_TOP_K            (default 20,  only used if sampling)

Usage:
  ADAPTER_DIR=/path/to/sof_dpo_ckpt CUDA_VISIBLE_DEVICES=0 \
    python script_parallel.py --model qwen3_vl_ft --shard-id 0 --num-shards 8
"""

import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
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


class Qwen3VLFinetunedEvaluator(BaseEvaluator):
    def load(self):
        adapter_dir = os.environ.get("ADAPTER_DIR", "")
        if not adapter_dir:
            raise ValueError(
                "ADAPTER_DIR env var is required for qwen3_vl_ft. "
                "Set it to the path of your LoRA adapter checkpoint."
            )
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"ADAPTER_DIR not found: {adapter_dir}")

        self.adapter_tag = os.environ.get(
            "ADAPTER_TAG", os.path.basename(adapter_dir.rstrip("/"))
        )

        # ─── Video preprocessing knobs (MUST match training!) ───────────────
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

        # Allow env override so the same loader works for 32B (or future variants)
        # without touching code; default keeps the original 8B behaviour.
        base_model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
        print(f"📦 Loading base model: {base_model_id}")
        print(f"🔌 Adapter: {adapter_dir} (tag: {self.adapter_tag})")
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

        self.processor = AutoProcessor.from_pretrained(base_model_id)

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            dtype=self.dtype,
            attn_implementation=config.ATTN_IMPL,
            device_map=self.device,
        )

        print(f"🔗 Merging adapter from {adapter_dir}...")
        model_with_adapter = PeftModel.from_pretrained(base_model, adapter_dir)

        # ── CRITICAL: load the trained full-rank weights for modules that
        # were excluded from LoRA (default: everything under `visual.*`,
        # i.e. the merger and deepstack mergers).  These weights live in
        # non_lora_state_dict.bin alongside the LoRA adapter and were trained
        # with `freeze_merger=False`.  Without this, the eval-time merger is
        # the stock initialization and ALL trained visual→language adaptation
        # is silently discarded — explains why "FT" runs sat near baseline.
        non_lora_path = os.path.join(adapter_dir, "non_lora_state_dict.bin")
        if os.path.isfile(non_lora_path):
            print(f"🔧 Loading trained non-LoRA weights: {non_lora_path}")
            nl_sd = torch.load(non_lora_path, map_location="cpu", weights_only=False)
            # Names in the file are prefixed with "base_model.model." because
            # they were captured after PeftModel wrapping; the unwrapped state
            # dict keys lack that prefix.  Try both forms.
            current_keys = {n for n, _ in model_with_adapter.named_parameters()}
            ren = {}
            for k, v in nl_sd.items():
                if k in current_keys:
                    ren[k] = v
                elif k.replace("base_model.model.", "") in current_keys:
                    ren[k.replace("base_model.model.", "")] = v
                else:
                    ren[k] = v  # let load_state_dict report it as unexpected
            missing, unexpected = model_with_adapter.load_state_dict(ren, strict=False)
            print(f"   loaded {len(ren)} tensors  "
                  f"(unexpected={len(unexpected)}, "
                  f"sample missing={list(missing)[:2] if missing else 'none'})")
            if unexpected:
                print(f"   ⚠️  unexpected keys (first 3): {list(unexpected)[:3]}")
        else:
            print(f"ℹ️  No non_lora_state_dict.bin in {adapter_dir} "
                  f"(merger will use base weights — OK only if freeze_merger=True at training time)")

        self.model = model_with_adapter.merge_and_unload()
        self.model.eval()
        print(f"✅ Qwen3-VL-8B + adapter + trained non-LoRA weights ready.")

    def _build_messages(self, video_path, prompt):
        # qwen-vl-utils reads these per-message overrides and supplies them to
        # the underlying frame sampler / smart_resize. They MUST be set, else
        # the library falls back to its (much larger) defaults — fps=2,
        # max_frames=768, max_pixels≈602k, total_pixels≈19.3M — completely
        # different distribution from training.
        # NOTE: SFT / DPO training data wraps the question as
        # "<video>\nQuestion: {q}". The chat template inserts the vision
        # tokens for us, so to land on the SAME post-template string we
        # only need to prepend "Question: " to the text content here.
        # Without this, every fine-tuned eval was running on a prompt
        # skeleton the model never saw at training time.
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

        proc_kwargs = {**video_kwargs}
        if video_metadata:
            proc_kwargs["video_metadata"] = video_metadata

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            do_resize=False,       # qwen-vl-utils already resized
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
