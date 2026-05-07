import torch
from typing import Optional

from models.base import BaseGenerator


class QwenVLGenerator(BaseGenerator):
    """
    Qwen VL model for local inference (supports Qwen2.5-VL, Qwen3-VL families).

    Set MODEL_ID to the HuggingFace model hub ID or a local checkpoint path.
    On a B200 / multi-GPU box this will shard automatically via device_map="auto".
    """

    # ------------------------------------------------------------------ #
    #  Knobs                                                               #
    # ------------------------------------------------------------------ #
    MODEL_ID: str = "Qwen/Qwen2.5-VL-72B-Instruct"  # ← swap here for Qwen3
    MAX_NEW_TOKENS: int = 1024
    # Pixels budget passed to Qwen's vision encoder per frame.
    # Lower → faster / less VRAM; higher → more detail.
    # 768*28*28 ≈ reasonable default for 72B on B200
    MIN_PIXELS: int = 256 * 28 * 28
    MAX_PIXELS: int = 1280 * 28 * 28

    def load(self):
        # Import here so the file is importable even without qwen deps installed
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
        )

        print(f"[QwenVL] Loading {self.MODEL_ID} …")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,          # bfloat16 by default
            attn_implementation="flash_attention_2",
            device_map="auto",               # auto-shards across all visible GPUs
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )
        print(f"[QwenVL] Model ready on {self.model.device}.")

    # ------------------------------------------------------------------ #
    #  Core generation                                                     #
    # ------------------------------------------------------------------ #
    def generate_response(
        self,
        prompt: str,
        video_path: Optional[str] = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        content: list[dict] = []

        if video_path is not None:
            content.append(
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": self.MIN_PIXELS,
                    "max_pixels": self.MAX_PIXELS,
                }
            )

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,       # greedy – stable for SFT data
            )

        # Trim the input tokens from the output
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


# ---------------------------------------------------------------------------
# Qwen3-VL subclass -- overrides everything that changed vs Qwen2.5-VL
# ---------------------------------------------------------------------------
class Qwen3VLGenerator(QwenVLGenerator):
    """
    Qwen3-VL family.
    Differences vs Qwen2.5-VL that are all overridden here:
      1. Model class  : Qwen3VLForConditionalGeneration
      2. Patch size   : 32x32 (Qwen2.5-VL used 28x28)
      3. video path   : requires file:// prefix
      4. process_vision_info : 3-value unpack with return_video_kwargs=True
         and video_metadata passed to processor (official Qwen3-VL pattern)
    """
    MODEL_ID: str = "Qwen/Qwen3-VL-32B-Instruct"

    # Qwen3-VL uses 32x32 patches, NOT 28x28 like Qwen2.5-VL
    MIN_PIXELS: int = 256 * 32 * 32
    MAX_PIXELS: int = 1280 * 32 * 32

    # fps fed into the video dict (official default is 2.0)
    VIDEO_FPS: float = 2.0

    def load(self):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        print(f"[Qwen3VL] Loading {self.MODEL_ID} ...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )
        print(f"[Qwen3VL] Model ready.")

    def generate_response(
        self,
        prompt: str,
        video_path: Optional[str] = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        content: list[dict] = []

        if video_path is not None:
            content.append(
                {
                    "type": "video",
                    "video": f"file://{video_path}",   # Qwen3-VL requires file:// prefix
                    "min_pixels": self.MIN_PIXELS,
                    "max_pixels": self.MAX_PIXELS,
                    "fps": self.VIDEO_FPS,
                }
            )

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Qwen3-VL: 3-value unpack — video_kwargs carries fps + metadata
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()