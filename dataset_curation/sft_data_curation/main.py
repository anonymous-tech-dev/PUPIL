"""
main.py -- CGBench -> SFT Data Generator
=========================================
All configurable knobs are in the section below.
Run:  python main.py

Model / Strategy compatibility
-------------------------------
Strategies 1 & 3 (text-only, no video frames):
    Best model  -> qwen3-32b        (Qwen/Qwen3-32B, fits on 1x B200, dense, best CoT)
    Alternative -> qwen3-30b-moe    (Qwen/Qwen3-30B-A3B-Instruct-2507, MoE, faster)
    Alternative -> qwen3.5-35b      (Qwen/Qwen3.5-35B-A3B, MoE)
    API option  -> gpt5 / gpt4o     (Azure Azure, requires az login)

Strategies 2 & 4 (video frames required):
    Best model  -> qwen3-vl-32b     (Qwen/Qwen3-VL-32B-Instruct, fits on 1x B200)
    API option  -> gpt5 / gpt4o     (Azure Azure, frames sampled client-side)
"""

# =========================================================================== #
#                         KNOBS -- EDIT HERE                                  #
# =========================================================================== #

# --- GPU selection -----------------------------------------------------------
# Which physical GPU(s) this process may use.
# Applied via os.environ before any CUDA context opens (done below).
#
# Single GPU   -> "0"        (text models like qwen3-32b fit on 1 GPU)
# All 4 GPUs   -> "0,1,2,3"  (for larger VL models)
# GPT / Azure  -> ""         (no GPU needed)
CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"

# --- Model -------------------------------------------------------------------
# Text-only (strategies 1 & 3):  "qwen3-32b" | "qwen3-30b-moe" | "qwen3.5-35b" | "qwen3.5-9b"
# Vision-language (all strats):  "qwen3-vl-32b" | "qwen2.5-7b" | "qwen2.5-72b"
# API / Azure (all strats):      "gpt5" | "gpt4o" | "gpt-Azure"
MODEL_KEY = "qwen3-vl-32b"

# Override the HuggingFace model ID or local checkpoint path.
# Set to None to use the default for MODEL_KEY.
MODEL_ID_OVERRIDE = None           # e.g. "/checkpoints/Qwen3-32B"

# Override the Azure Azure deployment name (GPT models only).
# Set to None to use the default for MODEL_KEY.
DEPLOYMENT_OVERRIDE = None         # e.g. "gpt-4o_2024-11-20"

# --- Strategy ----------------------------------------------------------------
# 1: transcript + Q + A           -> better_answer          (text-only)
# 2: transcript + Q + A + frames  -> better_answer          (vision required)
# 3: transcript + Q + A           -> reasoning_trace + better_answer  (text-only)
# 4: transcript + Q + A + frames  -> reasoning_trace + better_answer  (vision required)
STRATEGY = 4

# --- Paths -------------------------------------------------------------------
CGBENCH_JSON = "/workspace/Pupil/contrastive_experiments/cgbench_setup/cgbench.json"
CLUE_VID_DIR = "/data/Pupil/CGBench/clue_vids"
SUBTITLE_DIR = "/data/Pupil/CGBench/clue_vids_subtitles"
OUTPUT_DIR   = "/workspace/Pupil/dataset_curation/sft_data_curation/dahta"

# --- Index range -------------------------------------------------------------
# Slice of cgbench.json (0-based, END_IDX is exclusive).
#
# 4-GPU parallel split example (run one copy per terminal):
#   GPU 0: CUDA_VISIBLE_DEVICES="0", START_IDX=0,     END_IDX=3032
#   GPU 1: CUDA_VISIBLE_DEVICES="1", START_IDX=3032,  END_IDX=6064
#   GPU 2: CUDA_VISIBLE_DEVICES="2", START_IDX=6064,  END_IDX=9096
#   GPU 3: CUDA_VISIBLE_DEVICES="3", START_IDX=9096,  END_IDX=None
START_IDX = 0
END_IDX   = None  # exclusive; None = end of dataset

# --- Run settings ------------------------------------------------------------
MAX_RETRIES = 3   # retry attempts per item before skipping
SAVE_EVERY  = 10  # checkpoint after every N successfully generated items

# =========================================================================== #
#                         END OF KNOBS                                        #
# =========================================================================== #

import os
import sys
import json
import argparse
import signal
import subprocess
import math
from pathlib import Path
from datetime import datetime

# ---------- Multi-GPU launcher mode ----------
def _launch_multi_gpu():
    """Spawn one copy of this script per GPU, each processing a dataset shard."""
    import torch
    num_gpus = torch.cuda.device_count()
    with open(CGBENCH_JSON) as f:
        total = len(json.load(f))

    shard_size = math.ceil(total / num_gpus)
    procs = []
    for gpu_id in range(num_gpus):
        s = gpu_id * shard_size
        e = min(s + shard_size, total)
        if s >= total:
            break
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["_SFT_SHARD_START"] = str(s)
        env["_SFT_SHARD_END"] = str(e)
        env["_SFT_GPU_TAG"] = str(gpu_id)
        cmd = [sys.executable, __file__, "--_worker"]
        print(f"[launcher] GPU {gpu_id}: items [{s}, {e})  ({e - s} items)")
        p = subprocess.Popen(cmd, env=env)
        procs.append((gpu_id, p))

    print(f"\n[launcher] {len(procs)} workers running. Ctrl+C to stop all.\n")

    def _kill_all(signum=None, frame=None):
        print("\n[launcher] Caught interrupt, killing all workers…")
        for _, p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for _, p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("[launcher] All workers stopped.")
        sys.exit(1)

    signal.signal(signal.SIGINT, _kill_all)
    signal.signal(signal.SIGTERM, _kill_all)

    try:
        for gpu_id, p in procs:
            p.wait()
            status = "✓" if p.returncode == 0 else f"✗ (exit {p.returncode})"
            print(f"[launcher] GPU {gpu_id} finished {status}")
    except Exception:
        _kill_all()
    print("[launcher] All done.")
    sys.exit(0)


# ---------- Worker mode: override knobs from env ----------
if os.environ.get("_SFT_SHARD_START"):
    START_IDX = int(os.environ["_SFT_SHARD_START"])
    END_IDX = int(os.environ["_SFT_SHARD_END"])
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES)

# Apply GPU restriction BEFORE torch/transformers are imported
# (skip in launcher mode so it can see all GPUs)
if CUDA_VISIBLE_DEVICES != "" and "--multi_gpu" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES


def _build_output_path(total: int) -> str:
    """
    Auto-name the output file from the active knobs.
    Example: sft_strategy1_qwen3-32b_gpu0_idx0-3032_20250314_153022.json
    """
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = (MODEL_ID_OVERRIDE or MODEL_KEY).replace("/", "_").replace(" ", "_")
    gpu_label  = os.environ.get("_SFT_GPU_TAG", CUDA_VISIBLE_DEVICES)
    gpu_tag    = f"gpu{gpu_label.replace(',', '-')}" if gpu_label else "cpu"
    end        = END_IDX if END_IDX is not None else total
    idx_tag    = f"idx{START_IDX}-{end}"
    filename   = f"sft_strategy{STRATEGY}_{model_slug}_{gpu_tag}_{idx_tag}_{ts}.json"
    return str(Path(OUTPUT_DIR) / filename)


def main():
    print("=" * 62)
    print("  CGBench -> SFT Data Generator")
    print("=" * 62)
    print(f"  GPU(s)     : {CUDA_VISIBLE_DEVICES or '(none -- Azure/CPU)'}")
    print(f"  Model      : {MODEL_KEY}"
          + (f"  [override: {MODEL_ID_OVERRIDE}]" if MODEL_ID_OVERRIDE else ""))
    print(f"  Strategy   : {STRATEGY}")
    print(f"  Index range: [{START_IDX}, {END_IDX if END_IDX is not None else 'end'})")
    print(f"  CGBench    : {CGBENCH_JSON}")
    print(f"  Clue vids  : {CLUE_VID_DIR}")
    print(f"  Subtitles  : {SUBTITLE_DIR}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print("=" * 62)

    # Sanity-check input paths
    missing = [p for p in [CGBENCH_JSON, CLUE_VID_DIR, SUBTITLE_DIR]
               if not os.path.exists(p)]
    if missing:
        print("\n[error] The following paths do not exist:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    with open(CGBENCH_JSON) as f:
        total = len(json.load(f))

    print(f"\n  Dataset total : {total} items")
    print(f"  This shard    : items {START_IDX} -> {END_IDX if END_IDX is not None else total}")

    output_path = _build_output_path(total)
    print(f"  Output file   : {output_path}\n")

    # Load model -- strategy passed for early compatibility check
    from models.factory import build_generator
    generator = build_generator(
        model_key=MODEL_KEY,
        model_id_override=MODEL_ID_OVERRIDE,
        deployment_override=DEPLOYMENT_OVERRIDE,
        strategy=STRATEGY,     # validates text-only model not used with visual strategy
    )

    # Generate
    from data_generator import generate_sft_data
    generate_sft_data(
        cgbench_path=CGBENCH_JSON,
        clue_vid_dir=CLUE_VID_DIR,
        subtitle_dir=SUBTITLE_DIR,
        output_path=output_path,
        strategy=STRATEGY,
        generator=generator,
        max_retries=MAX_RETRIES,
        save_every=SAVE_EVERY,
        start_idx=START_IDX,
        end_idx=END_IDX,
    )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Launch one worker per GPU automatically")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.multi_gpu:
        _launch_multi_gpu()
    else:
        main()