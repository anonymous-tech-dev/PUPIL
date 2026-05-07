"""
Qwen3-VL-32B-Instruct on TRANSCRIPT only.

Same backbone as `qwen_32_vl.py`; differs only in the input modality (text
transcript instead of sampled video frames). On B200-class GPUs the BF16
weights (~64 GB) comfortably fit on a single device, so this works under
data-parallel sharding without model parallelism.
"""

from models.qwen3_vl_transcript import Qwen3VLTranscriptEvaluator


class Qwen32VLTranscriptEvaluator(Qwen3VLTranscriptEvaluator):
    MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
