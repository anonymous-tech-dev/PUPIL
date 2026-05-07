"""
main.py — Temporal Chain of Thought (TCoT) — Main Entry Point.

══════════════════════════════════════════════════════════════════════════════
  HOW TO USE
══════════════════════════════════════════════════════════════════════════════
1. Edit config.py to set your model, dataset, paths, and TCoT hyperparameters.
2. Run:  python main.py

All major settings live in config.py — there are no command-line argparse flags
by design (easier to track experiment configs in version control).

HOT-RESUME: Already-processed UIDs are detected from the results JSONL file
and skipped automatically.  Just re-run the script after an interruption.

══════════════════════════════════════════════════════════════════════════════
  PIPELINE STAGES
══════════════════════════════════════════════════════════════════════════════
  Stage 0 — Video Loading       (stages/stage0_video_loading.py)
             Decode video at 1 fps → list of (frame_id, PIL.Image)
             For dynamic_segment: metadata-only open + selective fetch.

  Stage 1 — Prompt Construction (stages/stage1_prompts.py)
             Build exact prompts from paper (Fig. 3, Fig. 14/15/16)

  Stage 2 — Selection Parsing   (stages/stage2_selection_parsing.py)
             Parse + validate JSON frame selection from model output

  Stage 3 — Context Aggregation (stages/stage3_context_aggregation.py)
             G(x, q): Single-Step / Dynamic-Segment / Hierarchical TCoT

  Stage 4 — Answering           (stages/stage4_answering.py)
             H(c, q) = f(c, q): pass curated context to VLM for final answer
"""

import logging
import os
import sys
import time
from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tcot.main")

# ─────────────────────────────────────────────────────────────────────────────
# All imports AFTER logging setup
# ─────────────────────────────────────────────────────────────────────────────

import torch
import random
import numpy as np

# ── Determinism: fix all seeds so re-runs produce identical results ────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import config
from models.factory import get_model
from utils.dataset_loaders import load_egoschema, load_lvbench
from utils.results_io import load_completed_uids, save_result, load_all_results
from stages.stage0_video_loading import load_video_frames
from stages.stage3_context_aggregation import aggregate_context
from stages.stage4_answering import answer_question

# ─────────────────────────────────────────────────────────────────────────────
# Dataset dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_iterator():
    n = config.NUM_SAMPLES
    if config.DATASET == "egoschema":
        return load_egoschema(num_samples=n)
    elif config.DATASET in ("lvbench_v1", "lvbench_v2"):
        return load_lvbench(num_samples=n)
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET!r}. "
                         "Choose 'egoschema', 'lvbench_v1', or 'lvbench_v2'.")
                         
# ─────────────────────────────────────────────────────────────────────────────
# Decide whether to use the fast (no-preload) path
# ─────────────────────────────────────────────────────────────────────────────

def _use_fast_path() -> bool:
    """
    The fast path (lazy frame decoding) applies when:
      - variant is dynamic_segment  (single_step / hierarchical need the full bundle)

    For single_step and hierarchical we still load all frames upfront because:
      - single_step needs full_bundle for _assemble_context uniform sampling
      - hierarchical needs full_bundle for neighbourhood expansion
    """
    return config.TCOT_VARIANT == "dynamic_segment"

# ─────────────────────────────────────────────────────────────────────────────
# Per-sample inference
# ─────────────────────────────────────────────────────────────────────────────

def run_sample(model, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    End-to-end TCoT inference for one QA sample.
    Returns a result dict ready to be saved to JSONL.
    """
    sample_start_time = time.time()

    uid            = item["uid"]
    video_path     = item["video_path"]
    question       = item["question"]
    answer_choices = item["answer_choices"]
    ground_truth   = item["ground_truth"]

    logger.info("Processing uid=%s | video=%s", uid, os.path.basename(video_path))

    fast = _use_fast_path()

    # ── Stage 0: Load frames (or skip for dynamic_segment fast path) ───────
    t0 = time.time()
    if fast:
        # Fast path: don't decode any frames yet — stage3 will do selective decode
        full_bundle  = None
        total_frames = None   # will be logged after stage3
        logger.info("  Stage 0 skipped (dynamic_segment fast path — lazy decode)")
    else:
        full_bundle  = load_video_frames(video_path, fps=config.VIDEO_FPS)
        total_frames = len(full_bundle)
        logger.info("  Stage 0 done: %d frames loaded (%.1fs)",
                    total_frames, time.time() - t0)

    # ── Stage 3: Context Aggregation (G function) ─────────────────────────
    t1 = time.time()
    agg = aggregate_context(
        model=model,
        full_bundle=full_bundle,
        question=question,
        answer_choices=answer_choices,
        variant=config.TCOT_VARIANT,
        video_path=video_path if fast else None,
    )

    # For fast path, learn total_frames from the context bundle + selection
    if total_frames is None:
        # We can infer it from the video metadata after the fact; use a sentinel
        # The results field will show -1 meaning "not pre-loaded"
        total_frames = -1

    logger.info("  Stage 3 done (%s): %d frames selected → %d in context (%.1fs)",
                agg["stage"],
                len(agg["selected_ids"]),
                len(agg["context_bundle"]),
                time.time() - t1)

    # ── Stage 4: Answering (H function) ───────────────────────────────────
    t2 = time.time()
    ans = answer_question(
        model=model,
        context_bundle=agg["context_bundle"],
        question=question,
        answer_choices=answer_choices,
    )
    logger.info("  Stage 4 done: predicted=%r  gt=%r (%.1fs)",
                ans["predicted_letter"], ground_truth, time.time() - t2)

    sample_total_time = time.time() - sample_start_time

    result = {
        "uid"              : uid,
        "predicted_letter" : ans["predicted_letter"],
        "ground_truth"     : ground_truth,
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
    }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    if cfg.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)
        
    # --- HYDRA MAGIC TRICK ---
    # Inject Hydra's YAML values into your existing config.py in memory
    for key, value in cfg.items():
        if key != "hydra": # skip the hydra system config
            setattr(config, key.upper(), value)
    # -------------------------

    logger.info("=" * 70)
    logger.info("Temporal Chain of Thought — DeepMind replication")
    logger.info("  Model   : %s", config.MODEL)
    logger.info("  Dataset : %s", config.DATASET)
    logger.info("  Variant : %s", config.TCOT_VARIANT)
    logger.info("  Segments: l=%d  s=%d  k=%d  u=%d",
                config.NUM_SEGMENTS, config.FRAMES_PER_SEGMENT,
                config.CONTEXT_BUDGET_FRAMES, config.UNIFORM_CONTEXT_FRAMES)
    logger.info("  Fast path (lazy decode): %s", _use_fast_path())
    logger.info("=" * 70)

    # ── Hot-resume: find already-done UIDs ────────────────────────────────
    completed = load_completed_uids()
    logger.info("Hot-resume: %d items already processed — will skip them.",
                len(completed))

    # ── Load model ────────────────────────────────────────────────────────
    model = get_model()
    model.load()

    # ── Main loop ─────────────────────────────────────────────────────────
    total   = 0
    correct = 0
    skipped = 0

    for item in get_dataset_iterator():
        uid = str(item["uid"])
        if uid in completed:
            skipped += 1
            continue

        # try:
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

        # except Exception as exc:
        #     logger.error("  ERROR on uid=%s: %s", uid, exc, exc_info=True)
        #     continue

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Run complete.  Processed=%d  Skipped=%d", total, skipped)
    if total > 0:
        logger.info("Final accuracy: %.2f%% (%d/%d)", 100.0 * correct / total,
                    correct, total)
    logger.info("Results saved to: %s", config.RESULTS_DIR)
    model.unload()

if __name__ == "__main__":
    main()