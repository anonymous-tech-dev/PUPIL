"""
InternVL3-78B evaluator — vanilla settings recommended by OpenGVLab.

Pipeline-identical to `InternVL3Evaluator` (the 8B variant in
`intern_3_vl.py`); only the checkpoint and the device-map strategy differ.

We use **InternVL3-78B** (not 3.5) because OpenGVLab never released a
78B-class InternVL3.5 checkpoint — the InternVL3.5 family tops out at 38B
dense (`InternVL3_5-38B`) and 241B sparse-MoE (`InternVL3_5-241B-A28B`).
So the 78B slot in our scaling sweep cleanly belongs to the InternVL3
series at its largest open size.

Reference: https://huggingface.co/OpenGVLab/InternVL3-78B
  • InternViT-6B vision encoder + Qwen2.5-72B LLM
  • 78B params bf16 ≈ 156GB → fits on a single 183GB B200 with KV-cache
    headroom (when GPU is otherwise idle).  We still pass
    `device_map="auto"` because (a) it lets HF auto-split if a fellow job
    is occupying part of the GPU, and (b) the OpenGVLab card explicitly
    recommends this path for >= 38B models.
"""

import math
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from models.intern_3_vl import InternVL3Evaluator


def _split_model_internvl3_78b(model_id: str):
    """Verbatim port of OpenGVLab's `split_model` from the InternVL3-78B
    model card. Pins vision_model, mlp1, embeddings, norm, lm_head, and the
    first/last LLM layers to GPU 0 to avoid cross-device tensor errors."""
    device_map = {}
    world_size = max(torch.cuda.device_count(), 1)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    num_layers = cfg.llm_config.num_hidden_layers
    # First GPU also holds ViT → treat it as half a GPU.
    denom = max(world_size - 0.5, 0.5)
    per = math.ceil(num_layers / denom)
    per_gpu = [per] * world_size
    per_gpu[0] = math.ceil(per_gpu[0] * 0.5)
    cnt = 0
    for i, n in enumerate(per_gpu):
        for _ in range(n):
            if cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{cnt}"] = i
            cnt += 1
    for k in [
        "vision_model",
        "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.output",
        "language_model.model.norm",
        "language_model.model.rotary_emb",
        "language_model.lm_head",
    ]:
        device_map[k] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


class InternVL3_78BEvaluator(InternVL3Evaluator):
    MODEL_ID = "OpenGVLab/InternVL3-78B"

    def load(self):
        # Use the OpenGVLab-recommended split_model device map (see model card).
        # This guarantees first/last LLM layers + ViT + embeddings + lm_head
        # all live on GPU 0, preventing the cross-device hidden-state errors
        # the card explicitly warns about with `device_map="auto"`.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True, use_fast=False
        )
        device_map = _split_model_internvl3_78b(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map=device_map,
        ).eval()
        # Resolve the actual device the input embedding lives on so we send
        # pixel_values to the same GPU (avoids cross-device errors that the
        # OpenGVLab card warns about — they recommend ensuring first/last LLM
        # layers + vision_model land on GPU 0).
        first_dev = next(self.model.parameters()).device
        self.device = str(first_dev)
        self.processor = self.tokenizer

    def generate_response(self, video_path, prompt):
        # 1. Sample frames (32 uniform — same as 8B)
        from models.intern_3_vl import load_video_frames, load_image_from_pil
        frames = load_video_frames(video_path, num_segments=self.NUM_FRAMES)

        # 2. Encode each frame as a single 448×448 tile (max_num=1).
        #    Place pixel_values on the SAME device as the first model layer.
        target_dev = next(self.model.parameters()).device
        pixel_values_list, num_patches_list = [], []
        for img in frames:
            pv = load_image_from_pil(img, input_size=448, max_num=1).to(torch.bfloat16).to(target_dev)
            num_patches_list.append(pv.size(0))
            pixel_values_list.append(pv)
        pixel_values = torch.cat(pixel_values_list, dim=0)

        # 3. "Frame{i+1}: <image>" prompt — verbatim from the OpenGVLab card.
        frame_prefix = "".join(f"Frame{i+1}: <image>\n" for i in range(len(frames)))
        question = frame_prefix + prompt

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
