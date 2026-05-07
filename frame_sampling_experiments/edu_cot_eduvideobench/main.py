"""
main.py — EduCoT × Pupil — Entry Point.

Hydra-based with optional shard slicing for data-parallel runs.

USAGE
─────
  # Smoke test (5 samples, base Qwen3-VL):
  python main.py num_samples=5 cuda_visible_devices=0

  # Fine-tuned LoRA adapter (single GPU):
  python main.py model.adapter_dir=/path/to/checkpoint-200 \\
                  model.adapter_tag=T04_gradfix_ckpt200

  # Sharded — usually launched via run_parallel.sh:
  python main.py shard_id=3 num_shards=8 cuda_visible_devices=3
"""

import logging
import os
import sys

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
    # Pin GPU before any CUDA import in submodules
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)

    import torch  # after CUDA_VISIBLE_DEVICES

    variant = build_variant_tag(cfg)

    shard_id   = int(cfg.get("shard_id", 0) or 0)
    num_shards = int(cfg.get("num_shards", 1) or 1)

    logger.info("=" * 70)
    logger.info("EduCoT — Education-Optimised Chain of Thought (Pupil)")
    logger.info("  Model         : %s", cfg.model.name)
    adapter_dir = cfg.model.get("adapter_dir") or os.environ.get("ADAPTER_DIR", "")
    adapter_tag = cfg.model.get("adapter_tag") or os.environ.get("ADAPTER_TAG", "")
    # If tag was not given but dir was, derive from basename so the SFT run is always namespaced.
    if adapter_dir and not adapter_tag:
        adapter_tag = os.path.basename(str(adapter_dir).rstrip("/"))
    # Hard sanity check: prevent silently overwriting base-model results.
    adapter_dir_s = (adapter_dir or "").strip()
    adapter_tag_s = (adapter_tag or "").strip()
    if adapter_dir_s:
        if not os.path.isdir(adapter_dir_s):
            logger.error("adapter_dir=%r does not exist or is not a directory. Refusing to run.", adapter_dir_s)
            sys.exit(2)
        if not adapter_tag_s:
            logger.error(
                "adapter_dir is set but adapter_tag could not be resolved. "
                "Refusing to run — results would silently overwrite the base-model file. "
                "Pass model.adapter_tag=<short_run_name> (or set env ADAPTER_TAG)."
            )
            sys.exit(2)
        # Push resolved values back into cfg so results_filepath() picks them up.
        cfg.model.adapter_dir = adapter_dir_s
        cfg.model.adapter_tag = adapter_tag_s
        # Defense in depth: the resolved results path MUST be SFT-namespaced.
        _path = results_filepath(cfg)
        if "_ft_" not in os.path.basename(_path):
            logger.error(
                "Adapter requested but resolved results path is NOT SFT-namespaced: %s. "
                "This would overwrite base-model results. Aborting.", _path,
            )
            sys.exit(2)
    logger.info("  Adapter dir   : %s", adapter_dir or "<none>")
    logger.info("  Adapter tag   : %s", adapter_tag or "<none>")
    logger.info("  Dataset       : %s", cfg.dataset.name)
    logger.info("  Segmentation  : %s", cfg.pipeline.segmentation)
    logger.info("  Keyframe filt : %s", cfg.pipeline.keyframe_filter)
    logger.info("  VLM selection : %s", cfg.pipeline.vlm_selection)
    logger.info("  Context budget: k=%d  u=%d",
                cfg.aggregation.context_budget_frames,
                cfg.aggregation.uniform_context_frames)
    logger.info("  Variant tag   : %s", variant)
    logger.info("  Shard         : %d / %d", shard_id, num_shards)
    logger.info("  Results file  : %s", results_filepath(cfg))
    logger.info("=" * 70)

    # ── Hot-resume ────────────────────────────────────────────────────────
    completed = load_completed_uids(cfg)
    logger.info("Hot-resume: %d items already done — skipping.", len(completed))

    # ── Materialise + shard-slice the dataset ─────────────────────────────
    all_items = list(get_dataset_iterator(cfg))
    all_items.sort(key=lambda x: str(x.get("uid", "")))
    items = all_items[shard_id::num_shards]
    logger.info("Dataset: %d total items → shard %d/%d gets %d items",
                len(all_items), shard_id, num_shards, len(items))

    # ── Load model ────────────────────────────────────────────────────────
    model = get_model(cfg)
    model.load()

    # ── Main loop ─────────────────────────────────────────────────────────
    total = correct = skipped = 0
    for item in items:
        uid = str(item["uid"])
        if uid in completed:
            skipped += 1
            continue

        try:
            result = run_pipeline(model, item, cfg)
            # Stamp shard info into each result for downstream merging.
            result["shard_id"]   = shard_id
            result["num_shards"] = num_shards
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
        logger.info("Final exact-match accuracy: %.2f%% (%d/%d)",
                    100.0 * correct / total, correct, total)
    logger.info("Results: %s", results_filepath(cfg))

    model.unload()


if __name__ == "__main__":
    main()
