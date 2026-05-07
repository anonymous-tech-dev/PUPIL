"""
config.py — Central configuration for Temporal Chain of Thought (TCoT)
All model/inference/dataset knobs are here. Edit this file to run experiments.
"""

# ─── Model Selection ───────────────────────────────────────────────────────────
# Options: "Qwen2.5-VL-7B"  |  "GPT-4o-mini"
MODEL = "Qwen2.5-VL-7B"

# ─── Dataset Selection ─────────────────────────────────────────────────────────
# Options: "egoschema"  |  "lvbench"
DATASET = "lvbench_v2"

# ─── Dataset Paths ─────────────────────────────────────────────────────────────
EGOSCHEMA_QUESTIONS = "/home/Pupil/frame_sampling_experiments/datasets/egoschema/subset_questions.json"
EGOSCHEMA_ANSWERS   = "/home/Pupil/frame_sampling_experiments/datasets/egoschema/subset_answers.json"
EGOSCHEMA_VIDEO_DIR = "/home/Pupil/frame_sampling_experiments/datasets/egoschema/videos/videos"

LVBENCH_V1_META        = "/home/Pupil/frame_sampling_experiments/datasets/LVBench/video_info.meta.jsonl"
LVBENCH_V1_VIDEO_DIR   = "/home/Pupil/frame_sampling_experiments/datasets/LVBench/videos"
# LVBENCH_V1_META      = "/workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/video_meta.jsonl"
# LVBENCH_V1_VIDEO_DIR = "/data/Pupil/lvbench_v1"
LVBENCH_V2_META      = "/workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/video_meta.jsonl"
LVBENCH_V2_VIDEO_DIR = "/data/Pupil/lvbench_v2"


# ─── Run Control ───────────────────────────────────────────────────────────────
# -1 = run on all samples. Set to a positive integer to run on N samples only.
NUM_SAMPLES = -1
CUDA_VISIBLE_DEVICES = "1"   # which physical GPU to use

# ─── TCoT Variant ──────────────────────────────────────────────────────────────
# Options: "single_step"  |  "dynamic_segment"  |  "hierarchical"
TCOT_VARIANT = "dynamic_segment"

# ─── Core TCoT Hyperparameters (§3.2 of paper) ────────────────────────────────
# Number of video segments (l). Higher → more compute, more coverage.
NUM_SEGMENTS = 12

# Frames sampled per segment for the selection call (s).
FRAMES_PER_SEGMENT = 64

# Total answering context budget in frames (k).
CONTEXT_BUDGET_FRAMES = 256

# Uniform context frames added alongside model-selected frames (u).
# These give the answerer global temporal awareness.
UNIFORM_CONTEXT_FRAMES = 0

# ─── Hierarchical TCoT (only used when TCOT_VARIANT == "hierarchical") ─────────
# Neighbourhood size (v): frames to expand around each selected frame.
HIER_NEIGHBOURHOOD = 10
# Max iterations before stopping.
HIER_MAX_ITERS = 3

# ─── Model Hyperparameters ─────────────────────────────────────────────────────
# Video sampling rate (frames per second) — paper uses 1 fps.
VIDEO_FPS = 1

# Max new tokens for the selection call (JSON output, so short).
SELECTION_MAX_TOKENS = 512
# Max new tokens for the answering call.
ANSWER_MAX_TOKENS = 512

# Generation temperature for answering call.
ANSWER_TEMPERATURE = 0.0

# ─── Qwen-VL specific ───────────────────────────────────────────────────────
# Set this to the model you want to run:
#   "Qwen/Qwen2.5-VL-7B-Instruct" for Qwen2.5-VL baseline/TCoT
#   "Qwen/Qwen3-VL-8B-Instruct"   for Qwen3-VL baseline/TCoT
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_DEVICE = "cuda"
ATTN_IMPL = "flash_attention_2"   # "eager" if flash-attn not available

# ─── GPT / Azure specific ──────────────────────────────────────────────────────
GPT_DEPLOYMENT = "gpt-4o_2024-11-20"
GPT_API_VERSION = "2024-10-21"
GPT_ENDPOINT = "https://<AZURE_OPENAI_ENDPOINT>"

# ─── Results ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"

# ─── Visualizer (visualizer.py only) ───────────────────────────────────────────
VIZ_OUTPUT_DIR = "viz_output"
VIZ_VIDEO_PATH = "/home/Pupil/dataset_curation/dataset/videos_db/final_1k/building_an_rc_raft_from_server_fans_and_insulation_board_clean.mp4"  # Set this to the video you want to visualize
VIZ_QUESTION    = "What is the color of the remote-controlled 4 wheel car?"  # The question to use during visualization
VIZ_ANSWER_CHOICES = ["Black and Pink", "White and Blue", "Black and White", "Red and Pink", "Black and Green"]  # List of 4–5 strings