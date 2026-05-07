"""
baseline.py — Standard (Baseline) VLM Inference — No TCoT Context Aggregation.

This is the direct comparison to TCoT described in the paper (§4.2, Tab. 1):

  'Baseline inference performs no context aggregation and answers the
   question directly.'

The model receives a uniformly-sampled set of frames (up to CONTEXT_BUDGET_FRAMES)
and is asked to answer the question directly — i.e. Stage 3 (context aggregation)
is skipped entirely.  This is equivalent to TCoT with the G function bypassed:

  a = f(x[k], q)       (standard inference, Eq. 1 in paper)

vs TCoT:

  c = G(x, q)          (context aggregation)
  a = H(c, q)          (answering on curated context)

Results are saved to:
  results/<dataset>/<model>_baseline_results.jsonl

so they sit alongside TCoT results and can be compared with evaluate.py.

HOT-RESUME: Already-processed UIDs are detected and skipped automatically.

══════════════════════════════════════════════════════════════════════════════
  KNOBS  (all inherited from config.py — edit that file)
══════════════════════════════════════════════════════════════════════════════

  MODEL                  — which VLM to use
  DATASET                — egoschema | lvbench
  NUM_SAMPLES            — -1 = all
  CONTEXT_BUDGET_FRAMES  — k  (how many frames to uniformly sample; paper uses 120)
  VIDEO_FPS              — sampling rate (paper uses 1 fps)
"""

import logging
import os
import sys
import time
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tcot.baseline")

import config
from models.factory import get_model
from utils.dataset_loaders import load_egoschema, load_lvbench
from utils.results_io import load_completed_uids, save_result
from stages.stage0_video_loading import load_video_frames, uniform_subsample
from stages.stage4_answering import answer_question

# Variant tag written to results — keeps baseline results separate from TCoT runs
BASELINE_VARIANT = "baseline"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset dispatcher (mirrors main.py)
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
# Per-sample baseline inference
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_sample(model, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard baseline inference for one QA sample.

    Steps:
      1. Decode video at VIDEO_FPS.
      2. Uniformly subsample CONTEXT_BUDGET_FRAMES frames  ← no selection call
      3. Pass directly to the answering call.

    Paper §3.1 / Tab. 1: 'Baseline inference … answers the question directly.'
    """
    sample_start_time = time.time()

    uid            = item["uid"]
    video_path     = item["video_path"]
    question       = item["question"]
    answer_choices = item["answer_choices"]
    ground_truth   = item["ground_truth"]

    logger.info("Baseline uid=%s | video=%s", uid, os.path.basename(video_path))

    # ── Stage 0: Load frames ───────────────────────────────────────────────
    t0 = time.time()
    full_bundle = load_video_frames(video_path, fps=config.VIDEO_FPS)
    logger.info("  Loaded %d frames (%.1fs)", len(full_bundle), time.time() - t0)

    # ── Uniform subsampling (NO selection call) ────────────────────────────
    k              = config.CONTEXT_BUDGET_FRAMES
    context_bundle = uniform_subsample(full_bundle, k)
    logger.info("  Uniformly sampled %d/%d frames (k=%d)",
                len(context_bundle), len(full_bundle), k)

    # ── Answering call ─────────────────────────────────────────────────────
    t1 = time.time()
    ans = answer_question(
        model=model,
        context_bundle=context_bundle,
        question=question,
        answer_choices=answer_choices,
    )
    logger.info("  Answer: predicted=%r  gt=%r (%.1fs)",
                ans["predicted_letter"], ground_truth, time.time() - t1)

    from stages.stage0_video_loading import get_frame_ids
    context_ids = get_frame_ids(context_bundle)
    sample_total_time = time.time() - sample_start_time

    result = {
        "uid"              : uid,
        "predicted_letter" : ans["predicted_letter"],
        "ground_truth"     : ground_truth,
        "raw_answer"       : ans["raw_response"],
        # Baseline has no selection — context_ids == all frames passed in
        "selected_ids"     : context_ids,
        "num_selected"     : len(context_ids),
        "context_ids"      : context_ids,
        "num_context"      : len(context_ids),
        "total_frames"     : len(full_bundle),
        "pct_selected"     : (
            100.0 * len(context_ids) / len(full_bundle)
            if full_bundle else 0.0
        ),
        # No selection justifications for baseline
        "justifications"   : [],
        "raw_responses"    : [],
        "stage"            : BASELINE_VARIANT,
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

def main():
    logger.info("=" * 70)
    logger.info("TCoT — Baseline Inference (no context aggregation)")
    logger.info("  Model              : %s", config.MODEL)
    logger.info("  Dataset            : %s", config.DATASET)
    logger.info("  Context frames (k) : %d", config.CONTEXT_BUDGET_FRAMES)
    logger.info("  Video FPS          : %s", config.VIDEO_FPS)
    logger.info("  Num samples        : %s",
                config.NUM_SAMPLES if config.NUM_SAMPLES != -1 else "all")
    logger.info("=" * 70)

    # ── Hot-resume ────────────────────────────────────────────────────────
    completed = load_completed_uids(variant=BASELINE_VARIANT)
    logger.info("Hot-resume: %d items already done — skipping.", len(completed))

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

        try:
            result = run_baseline_sample(model, item)
            save_result(result, variant=BASELINE_VARIANT)

            total += 1
            if (result["predicted_letter"]
                    and result["ground_truth"]
                    and result["predicted_letter"] == result["ground_truth"]):
                correct += 1

            acc = 100.0 * correct / total if total else 0.0
            logger.info("  Running accuracy: %.1f%% (%d/%d) [skipped=%d]",
                        acc, correct, total, skipped)

        except Exception as exc:
            logger.error("  ERROR on uid=%s: %s", uid, exc, exc_info=True)
            continue

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Baseline run complete.  Processed=%d  Skipped=%d", total, skipped)
    if total > 0:
        logger.info("Final baseline accuracy: %.2f%% (%d/%d)",
                    100.0 * correct / total, correct, total)
    logger.info("Results saved to: results/%s/ (variant=baseline)", config.DATASET)

    model.unload()


if __name__ == "__main__":
    main()