"""
main.py — Temporal Chain of Thought (TCoT) — Pupil Entry Point.

Argparse-based (no Hydra) so it composes cleanly with the data-parallel
launcher run_parallel.sh.

USAGE
─────
  # Single-GPU smoke test (5 samples, base Qwen3-VL):
  CUDA_VISIBLE_DEVICES=0 python main.py --max-samples 5

  # Single-GPU full run with fine-tuned LoRA adapter:
  CUDA_VISIBLE_DEVICES=0 ADAPTER_DIR=/path/to/checkpoint-200 \\
    ADAPTER_TAG=T04_gradfix_ckpt200 python main.py

  # Sharded (one shard per GPU) — usually launched by run_parallel.sh
  CUDA_VISIBLE_DEVICES=3 python main.py --shard-id 3 --num-shards 8

PIPELINE STAGES
───────────────
  Stage 0 — Video Loading       (stages/stage0_video_loading.py)
  Stage 3 — Context Aggregation (stages/stage3_context_aggregation.py)
  Stage 4 — Answering           (stages/stage4_answering.py)
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Any

# ── Logging setup BEFORE torch import ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tcot.main")

# ── Argparse FIRST so CUDA_VISIBLE_DEVICES can be honoured ─────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="TCoT on Pupil")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=-1,
                   help="Global cap on dataset size (applied BEFORE sharding, "
                        "so --num-samples N with --num-shards K yields ~N/K items per shard). -1 = all.")
    # Back-compat alias
    p.add_argument("--max-samples", dest="num_samples", type=int,
                   help=argparse.SUPPRESS)
    p.add_argument("--variant", type=str, default=None,
                   help="single_step | dynamic_segment | hierarchical")
    p.add_argument("--num-segments", type=int, default=None)
    p.add_argument("--frames-per-segment", type=int, default=None)
    p.add_argument("--context-budget", type=int, default=None)
    p.add_argument("--uniform-context", type=int, default=None)
    return p.parse_args()


_args = _parse_args()

# ── Now safe to import torch + config ──────────────────────────────────────
import torch
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import config

# Apply CLI overrides into config module
config.SHARD_ID   = _args.shard_id
config.NUM_SHARDS = _args.num_shards
if _args.variant is not None:
    config.TCOT_VARIANT = _args.variant
if _args.num_segments is not None:
    config.NUM_SEGMENTS = _args.num_segments
if _args.frames_per_segment is not None:
    config.FRAMES_PER_SEGMENT = _args.frames_per_segment
if _args.context_budget is not None:
    config.CONTEXT_BUDGET_FRAMES = _args.context_budget
if _args.uniform_context is not None:
    config.UNIFORM_CONTEXT_FRAMES = _args.uniform_context

from models.factory import get_model
from utils.dataset_loaders import load_egoschema, load_lvbench, load_Pupil
from utils.results_io import load_completed_uids, save_result, results_filepath
from stages.stage0_video_loading import load_video_frames
from stages.stage3_context_aggregation import aggregate_context
from stages.stage4_answering import answer_question


# ─────────────────────────────────────────────────────────────────────────────
# Dataset dispatcher (with shard slicing)
# ─────────────────────────────────────────────────────────────────────────────
def _all_items():
    name = config.DATASET.lower()
    if name == "egoschema":
        return list(load_egoschema(num_samples=-1))
    elif name in ("lvbench_v1", "lvbench_v2"):
        return list(load_lvbench(num_samples=-1))
    elif "eduvideo" in name or name == "edu":
        return list(load_Pupil(num_samples=-1))
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET!r}")


def get_dataset_iterator():
    items = _all_items()
    # deterministic ordering, then GLOBAL cap (so --num-samples N == N total
    # rows across all shards), then shard slicing.
    items.sort(key=lambda x: str(x.get("uid", "")))
    if _args.num_samples > 0:
        items = items[: _args.num_samples]
    items = items[config.SHARD_ID::config.NUM_SHARDS]
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Decide whether to use the fast (no-preload) path
# ─────────────────────────────────────────────────────────────────────────────
def _use_fast_path() -> bool:
    return config.TCOT_VARIANT == "dynamic_segment"


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample inference
# ─────────────────────────────────────────────────────────────────────────────
def run_sample(model, item: Dict[str, Any]) -> Dict[str, Any]:
    sample_start_time = time.time()

    uid            = item["uid"]
    video_path     = item["video_path"]
    question       = item["question"]
    answer_choices = item["answer_choices"]
    ground_truth   = item["ground_truth"]

    logger.info("Processing uid=%s | video=%s", uid, os.path.basename(video_path))

    fast = _use_fast_path()
    t0 = time.time()
    if fast:
        full_bundle  = None
        total_frames = None
        stage0_secs  = 0.0
        logger.info("  Stage 0 skipped (dynamic_segment fast path — lazy decode)")
    else:
        full_bundle  = load_video_frames(video_path, fps=config.VIDEO_FPS)
        total_frames = len(full_bundle)
        stage0_secs  = time.time() - t0
        logger.info("  Stage 0 done: %d frames loaded (%.1fs)",
                    total_frames, stage0_secs)

    t1 = time.time()
    agg = aggregate_context(
        model=model,
        full_bundle=full_bundle,
        question=question,
        answer_choices=answer_choices,
        variant=config.TCOT_VARIANT,
        video_path=video_path if fast else None,
    )
    if total_frames is None:
        total_frames = -1
    stage3_secs = time.time() - t1
    logger.info("  Stage 3 done (%s): %d frames selected → %d in context (%.1fs)",
                agg["stage"], len(agg["selected_ids"]),
                len(agg["context_bundle"]), stage3_secs)

    t2 = time.time()
    ans = answer_question(
        model=model,
        context_bundle=agg["context_bundle"],
        question=question,
        answer_choices=answer_choices,
    )
    stage4_secs = time.time() - t2
    logger.info("  Stage 4 done: predicted=%r  gt=%r (%.1fs)",
                ans["predicted_letter"], ground_truth, stage4_secs)

    sample_total_time = time.time() - sample_start_time

    return {
        # ── Judge-compatible schema (mirrors mllm_evaluation/script_parallel.py) ──
        "query_id"         : uid,
        "model_prediction" : ans["raw_response"],
        "ground_truth"     : ground_truth,
        "category"         : item.get("category", "unknown"),
        "source_of_fact"   : item.get("source_of_fact", "unknown"),
        "judge_verdict"    : None,
        "judge_reason"     : None,
        # ── Pipeline-specific extras ──
        "uid"              : uid,
        "predicted_letter" : ans["predicted_letter"],
        "raw_answer"       : ans["raw_response"],
        "selected_ids"     : agg["selected_ids"],
        "num_selected"     : len(agg["selected_ids"]),
        "context_ids"      : ans["frame_ids_used"],
        "num_context"      : len(ans["frame_ids_used"]),
        "total_frames"     : total_frames,
        "pct_selected"     : (
            100.0 * len(agg["selected_ids"]) / total_frames
            if total_frames and total_frames > 0 else -1.0
        ),
        "justifications"   : agg["justifications"],
        "raw_responses"    : agg["raw_responses"],
        "stage"            : agg["stage"],
        "video_path"       : video_path,
        "question"         : question,
        "answer_choices"   : answer_choices,
        "question_type"    : item.get("question_type", []),
        "time_reference"   : item.get("time_reference", ""),
        "time_taken_secs"  : round(sample_total_time, 2),
        # ── Provenance: lets you safely concatenate JSONLs from many SFT runs ──
        "model_name"       : config.MODEL,
        "adapter_dir"      : getattr(config, "ADAPTER_DIR", "") or "",
        "adapter_tag"      : getattr(config, "ADAPTER_TAG", "") or "",
        # ── Per-stage timings (seconds) for paper tables ──
        "timings"          : {
            "stage0_video_loading": round(stage0_secs, 2),
            "stage3_aggregation"  : round(stage3_secs, 2),
            "stage4_answering"    : round(stage4_secs, 2),
            "total"               : round(sample_total_time, 2),
        },
        # selection rounds + 1 final answering call
        "num_vlm_calls"    : len(agg["raw_responses"]) + 1,
        "shard_id"         : config.SHARD_ID,
        "num_shards"       : config.NUM_SHARDS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Hard sanity check: prevent silently overwriting base-model results ──
    _adapter_dir = (getattr(config, "ADAPTER_DIR", "") or "").strip()
    _adapter_tag = (getattr(config, "ADAPTER_TAG", "") or "").strip()
    if _adapter_dir:
        if not os.path.isdir(_adapter_dir):
            logger.error("ADAPTER_DIR=%r does not exist or is not a directory. Refusing to run.", _adapter_dir)
            sys.exit(2)
        if not _adapter_tag:
            logger.error(
                "ADAPTER_DIR is set but ADAPTER_TAG could not be resolved. "
                "Refusing to run — results would silently overwrite the base-model file. "
                "Set ADAPTER_TAG=<short_run_name> explicitly."
            )
            sys.exit(2)
        # Defense in depth: the resolved results path MUST be SFT-namespaced.
        _path = results_filepath()
        if "_ft_" not in os.path.basename(_path):
            logger.error(
                "Adapter requested but resolved results path is NOT SFT-namespaced: %s. "
                "This would overwrite base-model results. Aborting.", _path,
            )
            sys.exit(2)

    logger.info("=" * 70)
    logger.info("Temporal Chain of Thought — Pupil")
    logger.info("  Model        : %s", config.MODEL)
    logger.info("  Adapter tag  : %s", getattr(config, "ADAPTER_TAG", "") or "<none>")
    logger.info("  Adapter dir  : %s", getattr(config, "ADAPTER_DIR", "") or "<none>")
    logger.info("  Dataset      : %s", config.DATASET)
    logger.info("  Variant      : %s", config.TCOT_VARIANT)
    logger.info("  l/s/k/u      : %d / %d / %d / %d",
                config.NUM_SEGMENTS, config.FRAMES_PER_SEGMENT,
                config.CONTEXT_BUDGET_FRAMES, config.UNIFORM_CONTEXT_FRAMES)
    logger.info("  Shard        : %d / %d", config.SHARD_ID, config.NUM_SHARDS)
    logger.info("  Fast path    : %s", _use_fast_path())
    logger.info("  Results file : %s", results_filepath())
    logger.info("=" * 70)

    completed = load_completed_uids()
    logger.info("Hot-resume: %d items already processed — will skip them.", len(completed))

    items = get_dataset_iterator()
    logger.info("Shard %d/%d → %d items to process",
                config.SHARD_ID, config.NUM_SHARDS, len(items))

    model = get_model()
    model.load()

    total = correct = skipped = 0
    for item in items:
        uid = str(item["uid"])
        if uid in completed:
            skipped += 1
            continue
        try:
            result = run_sample(model, item)
            save_result(result)
            total += 1
            if (result["predicted_letter"]
                    and result["ground_truth"]
                    and result["predicted_letter"] == result["ground_truth"]):
                correct += 1
            acc = 100.0 * correct / total if total else 0.0
            logger.info("  Running accuracy: %.1f%% (%d/%d) [skipped=%d]",
                        acc, correct, total, skipped)
        except torch.cuda.OutOfMemoryError:
            logger.error("  OOM on uid=%s — skipping.", uid)
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            logger.error("  ERROR on uid=%s: %s", uid, exc, exc_info=True)
            continue

    logger.info("=" * 70)
    logger.info("Run complete. Processed=%d  Skipped=%d", total, skipped)
    if total > 0:
        logger.info("Final exact-match accuracy: %.2f%% (%d/%d)",
                    100.0 * correct / total, correct, total)
    logger.info("Results: %s", results_filepath())
    model.unload()


if __name__ == "__main__":
    main()
