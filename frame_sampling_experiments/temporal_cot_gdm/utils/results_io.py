"""
utils/results_io.py — Results Saving / Loading with Hot-Resume Support.

Results are stored as JSONL files:
  results/<dataset>/<model>_<variant>_<hparam_tag>_results.jsonl

Example filenames:
  Qwen2.5-VL-7B_dynamic_segment_l12_s64_k120_u32_results.jsonl
  Qwen2.5-VL-7B_baseline_k120_results.jsonl
  Qwen2.5-VL-7B_baseline_native_k120_results.jsonl
  Qwen2.5-VL-7B_seg_1_baseline_segs1_k120_results.jsonl

Each line is one completed QA item (see run_sample() for full schema).

Hot-resume: load_completed_uids() returns the set of UIDs already in the
results file so the main loop can skip them.
"""

import json
import os
from typing import Dict, Any, Set, Optional

import config


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter tag — embedded in every results filename
# ─────────────────────────────────────────────────────────────────────────────

def build_run_tag(variant: str = None, extra: Dict[str, Any] = None) -> str:
    """
    Build a compact hyperparameter tag for the results filename.

    TCoT variants:
      l{NUM_SEGMENTS}_s{FRAMES_PER_SEGMENT}_k{CONTEXT_BUDGET_FRAMES}_u{UNIFORM_CONTEXT_FRAMES}
      e.g. l12_s64_k120_u32

    Non-TCoT baselines (baseline, baseline_native):
      k{CONTEXT_BUDGET_FRAMES}
      e.g. k120

    Segment ablation (seg_N_baseline):
      segs{N}_k{CONTEXT_BUDGET_FRAMES}
      e.g. segs1_k120

    `extra` dict can inject additional key=value pairs.
    """
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

def _results_path(dataset=None, model=None, variant=None,
                  tag=None, extra=None):
    dataset = dataset or config.DATASET
    model   = (model or config.MODEL).replace("/", "-").replace(" ", "_")
    variant = variant or config.TCOT_VARIANT
    if tag is None:
        tag = build_run_tag(variant, extra)

    results_dir = os.path.join(
        os.path.dirname(__file__), "..", config.RESULTS_DIR, dataset
    )
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{model}_{variant}_{tag}_results_v7.jsonl"
    return os.path.join(results_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_completed_uids(dataset=None, model=None, variant=None,
                        tag=None, extra=None):
    """Return set of UIDs that have already been processed."""
    path = _results_path(dataset, model, variant, tag, extra)
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    done.add(str(obj["uid"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def save_result(result, dataset=None, model=None, variant=None,
                tag=None, extra=None):
    """Append one result dict as a JSONL line."""
    path = _results_path(dataset, model, variant, tag, extra)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_all_results(dataset=None, model=None, variant=None,
                     tag=None, extra=None):
    """Load all saved results as a list of dicts."""
    path = _results_path(dataset, model, variant, tag, extra)
    if not os.path.exists(path):
        return []
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def results_filepath(dataset=None, model=None, variant=None,
                     tag=None, extra=None):
    """Return the full path to the results file (useful for logging)."""
    return _results_path(dataset, model, variant, tag, extra)