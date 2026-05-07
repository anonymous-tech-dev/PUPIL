import torch
import os

# ==============================================================================
#                                     KNOBS
# ==============================================================================
# Environment variable overrides take FIRST PRIORITY.
#
# NOTE: Defaults below target the *current* trainer sandbox at /workspace/.
# Earlier runs lived under /home/Pupil/Pupil/... — that
# tree is still mounted in this sandbox, which would silently make hot-resume
# pick up stale results from a prior sandbox.  Always override
# EVAL_OUTPUT_DIR / EVAL_DATA_DIR / EVAL_VIDEO_DIR / EVAL_QUERY explicitly
# from the launcher to avoid that footgun.

_WORKSPACE_ROOT = "/workspace/Pupil"

base_data_dir = os.environ.get(
    "EVAL_DATA_DIR",
    os.path.join(_WORKSPACE_ROOT, "dataset_curation/dataset"),
)
# results_v2/ is the canonical output tree from May 5 2026 onward — keeps a
# clean separation from results/ (which holds older sandbox-local runs whose
# data paths/configs may differ).
base_output_dir = os.environ.get(
    "EVAL_OUTPUT_DIR",
    os.path.join(_WORKSPACE_ROOT, "mllm_evaluation/results_v2"),
)

VIDEO_DIR = os.environ.get(
    "EVAL_VIDEO_DIR",
    os.path.join(base_data_dir, "videos_db/final_1k"),
)

QUERY_FILE_PATH = os.environ.get(
    "EVAL_QUERY",
    os.path.join(base_data_dir, "queries_db/final_1k/final_consolidated_1k_final_v0.json"),
)

if not os.path.exists(QUERY_FILE_PATH):
    raise FileNotFoundError(
        f"CRITICAL: Query file not found at {QUERY_FILE_PATH}. "
        f"Set EVAL_QUERY env var or check your paths."
    )

DEFAULT_QUESTIONS = [
    "Describe this video in detail.",
    "What is the main activity happening here?"
]

OUTPUT_DIR = base_output_dir
OUTPUT_FOLDER = os.environ.get("EVAL_OUTPUT_FOLDER", "final_1k_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = os.environ.get("EVAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
ATTN_IMPL = "flash_attention_2"

# Models to evaluate in a single run (keys must match MODEL_REGISTRY in script.py)
MODELS_TO_EVALUATE = os.environ.get("EVAL_MODELS", "qwen3_vl").split(",")