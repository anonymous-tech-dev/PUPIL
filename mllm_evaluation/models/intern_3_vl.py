"""
InternVL3-8B evaluator — vanilla settings recommended by OpenGVLab.

Reference: https://huggingface.co/OpenGVLab/InternVL3-8B (model card)
  * `use_flash_attn=True`, `low_cpu_mem_usage=True`, bf16
  * Video preset: 32 uniformly sampled frames, each tile-encoded with
    `max_num=1` (1 patch per frame, 448x448), `use_thumbnail=True`
  * Prompt format: `Frame{i+1}: <image>\n` ... + question
  * Generation: `max_new_tokens=1024`, deterministic (`do_sample=False`)
    for benchmark reproducibility — the model card example shows
    `do_sample=True` for chat demos, but benchmarks (VLMEvalKit) use greedy.

See `intern_35_vl.py` for the InternVL3.5-8B variant (same pipeline).
"""

import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from models.base import BaseEvaluator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# -------- VLMEvalKit-equivalent image preprocessing --------
# Verbatim from temp_repo/VLMEvalKit/vlmeval/vlm/internvl/utils.py


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for r in target_ratios:
        ta = r[0] / r[1]
        d = abs(aspect_ratio - ta)
        if d < best_diff:
            best_diff, best = d, r
        elif d == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]:
            best = r
    return best


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    ow, oh = image.size
    aspect = ow / oh
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
                for i in range(1, n + 1) for j in range(1, n + 1)
                if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    target = find_closest_aspect_ratio(aspect, target_ratios, ow, oh, image_size)
    tw, th = image_size * target[0], image_size * target[1]
    blocks = target[0] * target[1]
    resized = image.resize((tw, th))
    out = []
    for i in range(blocks):
        box = (
            (i % (tw // image_size)) * image_size,
            (i // (tw // image_size)) * image_size,
            ((i % (tw // image_size)) + 1) * image_size,
            ((i // (tw // image_size)) + 1) * image_size,
        )
        out.append(resized.crop(box))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def load_image_from_pil(pil_img: Image.Image, input_size: int = 448, max_num: int = 1) -> torch.Tensor:
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(pil_img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


def load_video_frames(video_path: str, num_segments: int = 32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    indices = np.linspace(0, total - 1, num=num_segments, dtype=int)
    return [Image.fromarray(vr[idx].asnumpy()).convert("RGB") for idx in indices]


# ==========================================
# Evaluator
# ==========================================

class InternVL3Evaluator(BaseEvaluator):
    MODEL_ID = "OpenGVLab/InternVL3-8B"
    NUM_FRAMES = 32

    # ── OpenGVLab HF model card recommended generation settings ──
    # Greedy (do_sample=False) for benchmark reproducibility; max_new_tokens
    # 1024 per the model card's video chat example.
    GEN_KWARGS = dict(
        do_sample=False,
        max_new_tokens=1024,
        top_p=None,
    )

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map=self.device,
        ).eval()
        # Keep a processor handle present for BaseEvaluator.unload() compatibility
        self.processor = self.tokenizer

    def generate_response(self, video_path, prompt):
        # 1. Sample frames
        frames = load_video_frames(video_path, num_segments=self.NUM_FRAMES)

        # 2. Encode each frame as a single 448x448 tile (max_num=1)
        pixel_values_list, num_patches_list = [], []
        for img in frames:
            pv = load_image_from_pil(img, input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
            num_patches_list.append(pv.size(0))
            pixel_values_list.append(pv)
        pixel_values = torch.cat(pixel_values_list, dim=0)

        # 3. Build the "Frame{i+1}: <image>" prompt — verbatim from the
        #    OpenGVLab model card's video example (no hyphen).
        frame_prefix = "".join(f"Frame{i+1}: <image>\n" for i in range(len(frames)))
        question = frame_prefix + prompt

        # 4. Native InternVL chat API
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=question,
                generation_config=self.GEN_KWARGS,
                verbose=False,
            )
        return response
