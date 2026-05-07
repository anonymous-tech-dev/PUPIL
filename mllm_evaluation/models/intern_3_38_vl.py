"""
InternVL3-38B evaluator — vanilla settings recommended by OpenGVLab.

Pipeline-identical to `InternVL3Evaluator` (the 8B variant in
`intern_3_vl.py`); only the checkpoint changes.  Same 32-frame video preset,
`max_num=1` (1 patch per 448×448 frame), `Frame{i+1}: <image>` prompt
prefix, greedy decoding, `max_new_tokens=1024`.

Reference: https://huggingface.co/OpenGVLab/InternVL3-38B (model card)
  • InternViT-6B vision encoder + Qwen2.5-32B LLM
  • 38B params bf16 ≈ 76GB → fits a single B200 with KV-cache headroom
"""

from models.intern_3_vl import InternVL3Evaluator


class InternVL3_38BEvaluator(InternVL3Evaluator):
    MODEL_ID = "OpenGVLab/InternVL3-38B"
