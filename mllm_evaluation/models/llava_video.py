"""
LLaVA-Video-7B-Qwen2 evaluator.

Reference: lmms-lab/LLaVA-Video-7B-Qwen2 model card and
  https://github.com/LLaVA-VL/LLaVA-NeXT  (video usage example).

The official inference path uses the `llava` package's `load_pretrained_model`
helper to instantiate the LLaVA wrapper around the underlying Qwen2-7B LM and
SigLIP vision tower.  We mirror that here.

Frame budget: 32 (paper default for video QA evals).  Greedy decoding.
"""

import torch
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from models.base import BaseEvaluator


def _patch_transformers_for_llava():
    """LLaVA-NeXT imports several symbols from `transformers.modeling_utils`
    that recent transformers (>=4.50) only expose under `transformers.pytorch_utils`.
    Re-export them at the legacy location BEFORE any `import llava` happens."""
    import transformers.modeling_utils as _mu
    from transformers import pytorch_utils as _pu
    for _name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(_mu, _name) and hasattr(_pu, _name):
            setattr(_mu, _name, getattr(_pu, _name))


# Apply the patch eagerly at module import — must happen before `import llava`.
_patch_transformers_for_llava()


class LLaVAVideoEvaluator(BaseEvaluator):
    MODEL_ID = "lmms-lab/LLaVA-Video-7B-Qwen2"
    NUM_FRAMES = 32
    MAX_NEW_TOKENS = 4096

    # LLaVA-NeXT-Video conv template (as in the official README example)
    CONV_TEMPLATE = "qwen_1_5"

    def load(self):
        # Lazy import — `llava` (LLaVA-NeXT) is an optional heavy dep.
        _patch_transformers_for_llava()
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = "llava_qwen"
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            self.MODEL_ID,
            None,                         # model_base
            model_name,
            torch_dtype="bfloat16",
            device_map=self.device,
            attn_implementation="sdpa",
        )
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.max_length = max_length
        # BaseEvaluator.unload() expects self.processor
        self.processor = image_processor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_frames(self, video_path: str):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total = len(vr)
        idxs = np.linspace(0, total - 1, num=self.NUM_FRAMES, dtype=int)
        # Return numpy array (T, H, W, 3) uint8 — what LLaVA's image_processor expects
        return vr.get_batch(idxs).asnumpy()

    def _build_conv_prompt(self, question: str) -> str:
        """Build a single-turn user prompt with the video token, mirroring
        the official LLaVA-NeXT-Video example for `qwen_1_5` template."""
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        time_instruction = (
            "The video lasts for several seconds, "
            "and 32 frames are uniformly sampled from it.\n"
        )
        conv = conv_templates[self.CONV_TEMPLATE].copy()
        conv.append_message(
            conv.roles[0],
            f"{DEFAULT_IMAGE_TOKEN}\n{time_instruction}{question}",
        )
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def generate_response(self, video_path: str, prompt: str) -> str:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        frames_np = self._sample_frames(video_path)            # (T, H, W, 3)
        # image_processor.preprocess returns dict with "pixel_values" tensor (T, 3, H', W')
        video_tensor = self.image_processor.preprocess(
            frames_np, return_tensors="pt"
        )["pixel_values"].to(self.device, dtype=torch.bfloat16)

        prompt_text = self._build_conv_prompt(prompt)
        input_ids = (
            tokenizer_image_token(
                prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                do_sample=False,
                max_new_tokens=self.MAX_NEW_TOKENS,
            )
        text = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        return text
