"""
config.py — TCoT configuration for Pupil evaluation.

Copied from temporal_cot_gdm. Changes:
  - ADAPTER_DIR support via env var (LoRA fine-tuned Qwen3-VL)
  - Open-ended QA dataset (Pupil)
  - Pupil paths
"""
import os

# ─── Model Selection ─────────────────────────────────────────────────────────
MODEL = "Qwen3-VL-8B"

# ─── Dataset Selection ───────────────────────────────────────────────────────
DATASET = "Pupil"

# ─── Pupil paths ─────────────────────────────────────────────────────
Pupil_META = "/workspace/Pupil/dataset_curation/dataset/queries_db/final_1k/final_1k_for_cot.jsonl"
Pupil_VIDEO_DIR = "/workspace/Pupil/dataset_curation/dataset/videos_db/final_1k"

# ─── (legacy) LVBench / Egoschema paths — kept to avoid import errors ────────
LVBENCH_V1_META      = ""
LVBENCH_V1_VIDEO_DIR = ""
LVBENCH_V2_META      = "/workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/video_meta.jsonl"
LVBENCH_V2_VIDEO_DIR = "/data/Pupil/lvbench_v2"
EGOSCHEMA_QUESTIONS = ""
EGOSCHEMA_ANSWERS   = ""
EGOSCHEMA_VIDEO_DIR = ""

# ─── Run Control ─────────────────────────────────────────────────────────────
NUM_SAMPLES = -1
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# ─── Sharding (set by main.py from CLI) ──────────────────────────────────────
SHARD_ID   = 0
NUM_SHARDS = 1

# ─── TCoT Variant ────────────────────────────────────────────────────────────
TCOT_VARIANT = "dynamic_segment"

# ─── Core TCoT Hyperparameters (§3.2) ────────────────────────────────────────
NUM_SEGMENTS            = 12
FRAMES_PER_SEGMENT      = 128
CONTEXT_BUDGET_FRAMES   = 512
UNIFORM_CONTEXT_FRAMES  = 128

# ─── Hierarchical TCoT ───────────────────────────────────────────────────────
HIER_NEIGHBOURHOOD = 10
HIER_MAX_ITERS = 3

# ─── Model Hyperparameters ───────────────────────────────────────────────────
VIDEO_FPS = 1
SELECTION_MAX_TOKENS = 512
ANSWER_MAX_TOKENS = 512
ANSWER_TEMPERATURE = 0.0

# ─── Qwen-VL specific ────────────────────────────────────────────────────────
QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_DEVICE = "cuda"
ATTN_IMPL = "flash_attention_2"

# ─── LoRA Adapter (for fine-tuned Qwen3-VL) ──────────────────────────────────
# When ADAPTER_DIR env var is set, qwen3_vl loader merges this LoRA on top of base.
#
# IMPORTANT: We treat empty-string env vars the same as "unset". The parallel
# launcher always exports ADAPTER_DIR / ADAPTER_TAG (even when empty), so a
# naive os.environ.get(name, default) would return "" instead of the default
# and silently produce base-model output filenames — overwriting prior runs.
ADAPTER_DIR = (os.environ.get("ADAPTER_DIR") or "").strip()
_env_tag = (os.environ.get("ADAPTER_TAG") or "").strip()
ADAPTER_TAG = _env_tag or (
    os.path.basename(ADAPTER_DIR.rstrip("/")) if ADAPTER_DIR else ""
)

# ─── GPT / Azure ─────────────────────────────────────────────────────────────
GPT_DEPLOYMENT = "gpt-5.1_2025-11-13"
GPT_API_VERSION = "2024-10-21"
GPT_ENDPOINT = "https://<AZURE_OPENAI_ENDPOINT>"

# ─── Results ─────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
VIZ_OUTPUT_DIR = "viz_output"
