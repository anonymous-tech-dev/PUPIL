"""
Qwen2.5-VL-32B-Instruct evaluator.

Pipeline-identical to `Qwen2_5_VLEvaluator` (the 7B variant in
`qwen_25_vl.py`); only the checkpoint and `device_map` differ.  We keep the
exact same VLMEvalKit "ForVideo" pixel budgets + generation hyperparameters
so the 32B vs 7B delta is purely model-size / training, not preprocessing.

Reference: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
"""

from models.qwen_25_vl import Qwen2_5_VLEvaluator


class Qwen2_5_VL_32BEvaluator(Qwen2_5_VLEvaluator):
    MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

    # 32B in bf16 ≈ 64GB → fits a single B200 (183GB) comfortably with
    # video-KV headroom.  Still pass `device_map=self.device` (single GPU
    # per shard) so 8 shards × 8 GPUs work out of the box.
