"""
Oryx-1.5-32B (THUdyh/Oryx-1.5-32B) evaluator.

Oryx-1.5-32B is a Qwen2.5-32B–based multimodal LLM with the OryxViT
(SigLIP-SO400M @ patch16/384) vision tower and a dynamic-compressor
resampler. We follow the author-recommended inference pipeline from
https://github.com/Oryx-mllm/Oryx (`inference.py` and the model card on
https://huggingface.co/THUdyh/Oryx-1.5-32B):

  * conv template:  qwen_1_5
  * frames:         64 uniformly-sampled (force_sample), per the model card
  * preprocess:     process_anyres_video_genli with image_processor.do_resize
                    and do_center_crop disabled
  * tokenization:   tokenizer_image_token  (the `34b/32b` path in inference.py
                    — the `7b` branch uses the LLaVA-style preprocess_qwen,
                    but 32B uses the simpler tokenizer_image_token route)
  * stopping:       KeywordsStoppingCriteria on conv.sep / conv.sep2

The vision tower is auto-downloaded from THUdyh/Oryx-ViT on first load
(see oryx/model/multimodal_encoder/oryx_vit.py:create_siglip_vit), so no
manual config.json patching is required.

NOTE on routing in oryx.model.builder.load_pretrained_model:
The upstream builder branches on `"7b" in model_name.lower()` to select the
OryxQwen vs OryxLlama path.  For a Qwen2.5-based 32B checkpoint we MUST take
the OryxQwen path, so we pass `model_name="oryx_qwen_7b"` purely as a routing
hack (it has no other side effect; the model code is shared and the actual
weights come from the checkpoint, which already declares
architectures=["OryxQwenForCausalLM"]).
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

from models.base import BaseEvaluator

# Make sure the cloned `third_party/oryx` package is importable even if it
# wasn't `pip install -e .`'d (defensive — we install it in setup, but this
# keeps the evaluator self-contained).
_ORYX_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "oryx")
)
if os.path.isdir(_ORYX_REPO) and _ORYX_REPO not in sys.path:
    sys.path.insert(0, _ORYX_REPO)


# ─── Author-recommended preprocessing env vars (mirrors inference.sh /
# scripts/eval_video.sh from the Oryx repo).  These MUST be set BEFORE
# `import oryx.mm_utils`, since that module reads them at import time. ───
os.environ.setdefault("LOWRES_RESIZE", "384x32")
os.environ.setdefault("VIDEO_RESIZE", "0x64")
os.environ.setdefault("HIGHRES_BASE", "0x32")
os.environ.setdefault("MAXRES", "1536")
os.environ.setdefault("MINRES", "0")
os.environ.setdefault("VIDEO_MAXRES", "480")
os.environ.setdefault("VIDEO_MINRES", "288")
os.environ.setdefault("PAD2STRIDE", "1")


def _patch_transformers_for_oryx():
    """Oryx's qformer.py imports a few helpers from `transformers.modeling_utils`
    that recent transformers (>=4.50) only expose under
    `transformers.pytorch_utils`. Re-export them at the legacy location BEFORE
    any `import oryx` happens. Mirrors the same shim used in models/llava_video.py.
    """
    import transformers.modeling_utils as _mu
    from transformers import pytorch_utils as _pu
    for _name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(_mu, _name) and hasattr(_pu, _name):
            setattr(_mu, _name, getattr(_pu, _name))


_patch_transformers_for_oryx()


class Oryx15_32BEvaluator(BaseEvaluator):
    MODEL_ID = "THUdyh/Oryx-1.5-32B"
    NUM_FRAMES = 64
    CONV_TEMPLATE = "qwen_1_5"

    GEN_KWARGS = dict(
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=1,
        max_new_tokens=1024,
        use_cache=True,
    )

    def load(self):
        from oryx.model.builder import load_pretrained_model
        from oryx.utils import disable_torch_init

        disable_torch_init()

        # Match inference.py: enable dynamic compressor + sdpa.
        overwrite_config = {
            "mm_resampler_type": "dynamic_compressor",
            "patchify_video_feature": False,
            "attn_implementation": "sdpa" if torch.__version__ >= "2.1.2" else "eager",
        }

        # See class docstring: route through the OryxQwen branch.
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.MODEL_ID,
            None,                       # model_base
            "oryx_qwen_7b",              # routing hack → OryxQwenForCausalLM
            device_map=self.device,
            overwrite_config=overwrite_config,
        )
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        # BaseEvaluator.unload() expects self.processor.
        self.processor = image_processor

    # ------------------------------------------------------------------
    def _sample_frames(self, video_path: str):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total = len(vr)
        idxs = np.linspace(0, total - 1, num=self.NUM_FRAMES, dtype=int).tolist()
        arr = vr.get_batch(idxs).asnumpy()                # (T, H, W, 3) uint8
        return [Image.fromarray(f) for f in arr]

    # ------------------------------------------------------------------
    def generate_response(self, video_path: str, prompt: str) -> str:
        from oryx.conversation import conv_templates, SeparatorStyle
        from oryx.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from oryx.mm_utils import (
            tokenizer_image_token,
            KeywordsStoppingCriteria,
            process_anyres_video_genli,
        )

        frames = self._sample_frames(video_path)

        question = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        conv = conv_templates[self.CONV_TEMPLATE].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        # Per the official inference script: disable resize / center-crop on
        # the image_processor before calling process_anyres_video_genli.
        self.image_processor.do_resize = False
        self.image_processor.do_center_crop = False

        video_processed = []
        for frame in frames:
            f = process_anyres_video_genli(frame, self.image_processor)
            video_processed.append(f.unsqueeze(0))
        video_processed = (
            torch.cat(video_processed, dim=0).bfloat16().to(self.model.device)
        )
        # Oryx expects a (lowres, highres) tuple; same tensor passes both.
        video_data = (video_processed, video_processed)

        stop_str = (
            conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        )
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=video_data[0],
                images_highres=video_data[1],
                modalities="video",
                stopping_criteria=[stopping_criteria],
                **self.GEN_KWARGS,
            )

        text = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()
        return text
