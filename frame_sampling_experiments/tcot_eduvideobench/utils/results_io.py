"""
utils/results_io.py — Results saving / loading with hot-resume support.

Filename layout (Pupil edition):
  results/<dataset>/<model>[_ft_<adapter_tag>]_<variant>_<tag>[_shardN]_results.jsonl

Example:
  results/Pupil/Qwen3-VL-8B_dynamic_segment_l12_s128_k512_u128_results.jsonl
  results/Pupil/Qwen3-VL-8B_ft_checkpoint-200_dynamic_segment_l12_s128_k512_u128_shard3of8_results.jsonl

Each JSONL line is one completed QA item. `load_completed_uids` returns the
set of UIDs already processed for hot-resume.
"""

import json
import os
from typing import Dict, Any, Set, Optional

import config


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter tag
# ─────────────────────────────────────────────────────────────────────────────

def build_run_tag(variant: str = None, extra: Dict[str, Any] = None) -> str:
    v = variant or config.TCOT_VARIANT

    if v in ("baseline", "baseline_native"):
        tag = f"k{config.CONTEXT_BUDGET_FRAMES}"
    elif "seg_" in v and "baseline" in v:
        segs = (extra or {}).get("segs", "?")
        tag  = f"segs{segs}_k{config.CONTEXT_BUDGET_FRAMES}"
    else:
        tag = (
            f"l{config.NUM_SEGMENTS}"
            f"_s{config.FRAMES_PER_SEGMENT}"
            f"_k{config.CONTEXT_BUDGET_FRAMES}"
            f"_u{config.UNIFORM_CONTEXT_FRAMES}"
        )

    if extra:
        for k, val in extra.items():
            if k != "segs":
                tag += f"_{k}{val}"

    return tag


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_tag() -> str:
    """Model name + optional adapter tag for directory / file naming."""
    base = config.MODEL.replace("/", "-").replace(" ", "_")
    adapter_tag = getattr(config, "ADAPTER_TAG", "") or ""
    if adapter_tag:
        return f"{base}_ft_{adapter_tag}"
    return base


def _shard_suffix() -> str:
    n = getattr(config, "NUM_SHARDS", 1)
    if n is None or n <= 1:
        return ""
    return f"_shard{getattr(config, 'SHARD_ID', 0)}of{n}"


def _results_path(dataset=None, model=None, variant=None,
                  tag=None, extra=None):
    dataset = dataset or config.DATASET
    model   = model or _model_tag()
    variant = variant or config.TCOT_VARIANT
    if tag is None:
        tag = build_run_tag(variant, extra)

    results_dir = os.path.join(
        os.path.dirname(__file__), "..", config.RESULTS_DIR, dataset
    )
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{model}_{variant}_{tag}{_shard_suffix()}_results.jsonl"
    return os.path.join(results_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_completed_uids(dataset=None, model=None, variant=None,
                        tag=None, extra=None) -> Set[str]:
    path = _results_path(dataset, model, variant, tag, extra)
    if not os.path.exists(path):
        return set()
    done: Set[str] = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["uid"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def save_result(result, dataset=None, model=None, variant=None,
                tag=None, extra=None) -> None:
    path = _results_path(dataset, model, variant, tag, extra)
    with open(path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def load_all_results(dataset=None, model=None, variant=None,
                     tag=None, extra=None):
    path = _results_path(dataset, model, variant, tag, extra)
    if not os.path.exists(path):
        return []
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def results_filepath(dataset=None, model=None, variant=None,
                     tag=None, extra=None) -> str:
    return _results_path(dataset, model, variant, tag, extra)
