"""
main.py — EduCoT: Education-Optimised Chain of Thought — Entry Point.

═══════════════════════════════════════════════════════════════════════════
  USAGE
═══════════════════════════════════════════════════════════════════════════

  # Default config (uniform segments + VLM selection, Qwen3-VL)
  python main.py

  # Baseline: uniform sample k frames → answer (no selection)
  python main.py pipeline.vlm_selection=false

  # TCoT replication
  python main.py pipeline.segmentation=uniform pipeline.vlm_selection=true

  # Full EduCoT: scene-detect + keyframe filter + VLM selection
  python main.py pipeline.segmentation=scene_detect pipeline.keyframe_filter=true

  # Switch model
  python main.py model.name=Qwen2.5-VL-7B model.model_id=Qwen/Qwen2.5-VL-7B-Instruct

  # Adjust context budget
  python main.py aggregation.context_budget_frames=512

  # Switch GPU
  python main.py cuda_visible_devices=2

═══════════════════════════════════════════════════════════════════════════
"""

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("educot.main")

import hydra
from omegaconf import DictConfig

from pipeline import run_pipeline
from models.factory import get_model
from utils.dataset_loaders import get_dataset_iterator
from utils.results_io import (
    load_completed_uids, save_result, results_filepath, build_variant_tag,
)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Pin GPU before any CUDA import
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)

    import torch  # noqa: import after CUDA_VISIBLE_DEVICES is set

    variant = build_variant_tag(cfg)

    logger.info("=" * 70)
    logger.info("EduCoT — Education-Optimised Chain of Thought")
    logger.info("  Model         : %s", cfg.model.name)
    logger.info("  Dataset       : %s", cfg.dataset.name)
    logger.info("  Segmentation  : %s", cfg.pipeline.segmentation)
    logger.info("  Keyframe filt : %s", cfg.pipeline.keyframe_filter)
    logger.info("  VLM selection : %s", cfg.pipeline.vlm_selection)
    logger.info("  Context budget: k=%d  u=%d",
                cfg.aggregation.context_budget_frames,
                cfg.aggregation.uniform_context_frames)
    logger.info("  Variant tag   : %s", variant)
    logger.info("  Results file  : %s", results_filepath(cfg))
    logger.info("=" * 70)

    # ── Hot-resume ────────────────────────────────────────────────────────
    completed = load_completed_uids(cfg)
    logger.info("Hot-resume: %d items already done — skipping.", len(completed))

    # ── Load model ────────────────────────────────────────────────────────
    model = get_model(cfg)
    model.load()

    # ── Main loop ─────────────────────────────────────────────────────────
    total = 0
    correct = 0
    skipped = 0

    for item in get_dataset_iterator(cfg):
        uid = str(item["uid"])
        if uid in completed:
            skipped += 1
            continue

        try:
            result = run_pipeline(model, item, cfg)
            save_result(result, cfg)

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

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Done.  Processed=%d  Skipped=%d", total, skipped)
    if total > 0:
        logger.info("Final accuracy: %.2f%% (%d/%d)",
                    100.0 * correct / total, correct, total)
    logger.info("Results: %s", results_filepath(cfg))

    model.unload()


if __name__ == "__main__":
    main()
