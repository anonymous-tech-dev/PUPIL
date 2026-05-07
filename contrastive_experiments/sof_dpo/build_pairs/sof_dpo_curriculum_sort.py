"""
sof_dpo_curriculum_sort.py — Re-order DPO / SFT rows easy → hard for a
curriculum-learning training run.

Difficulty proxy = (1 - baseline_acc[(axis, cognitive_category)]) computed
from the qwen3_vl baseline run on the benchmark. Cells with no baseline
data fall back to (1 - per-axis baseline_acc) which falls back to 0.5.

Usage
-----
    python sof_dpo_curriculum_sort.py \
        --in-train ../old_dpo_revised_data_8b/sof_dpo_train.judged.json \
        --in-sft   ../old_dpo_revised_data_8b/sof_sft_warmstart.no_transcript.json \
        --out-train ../old_dpo_revised_data_8b/sof_dpo_train.judged.curriculum.json \
        --out-sft   ../old_dpo_revised_data_8b/sof_sft_warmstart.no_transcript.curriculum.json \
        --baseline-results-dir /workspace/Pupil/mllm_evaluation/results/qwen3_vl/final_1k_benchmark
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path


def load_baseline_acc(results_dir: str) -> tuple[dict, dict]:
    """Return (per_pair_acc, per_axis_acc) where keys are (axis,cat) / axis."""
    rows = []
    for f in glob.glob(os.path.join(results_dir, "*_results.json")):
        try:
            rows += json.load(open(f))
        except Exception:
            pass
    rows = [r for r in rows if isinstance(r, dict)
            and r.get("judge_verdict") is not None]
    pair = defaultdict(lambda: [0, 0])
    ax = defaultdict(lambda: [0, 0])
    for r in rows:
        a = r.get("source_of_fact", "?")
        c = r.get("category", "?")
        pair[(a, c)][1] += 1
        ax[a][1] += 1
        if r.get("judge_verdict") is True:
            pair[(a, c)][0] += 1
            ax[a][0] += 1
    pair_acc = {k: v[0] / max(1, v[1]) for k, v in pair.items()}
    ax_acc = {k: v[0] / max(1, v[1]) for k, v in ax.items()}
    return pair_acc, ax_acc


def difficulty(record: dict, pair_acc: dict, ax_acc: dict,
                axis_key: str, cat_key: str) -> float:
    a = record.get(axis_key, "?")
    c = record.get(cat_key, "?")
    if (a, c) in pair_acc:
        return 1.0 - pair_acc[(a, c)]
    if a in ax_acc:
        return 1.0 - ax_acc[a]
    return 0.5


def sort_one(rows: list[dict], pair_acc, ax_acc,
             axis_key: str, cat_key: str) -> list[dict]:
    # Stable sort; secondary key by id for determinism.
    return sorted(
        rows,
        key=lambda r: (
            difficulty(r, pair_acc, ax_acc, axis_key, cat_key),
            r.get("id") or r.get("query_id") or "",
        ),
    )


def report(name: str, rows: list[dict], pair_acc, ax_acc,
           axis_key: str, cat_key: str):
    diffs = [difficulty(r, pair_acc, ax_acc, axis_key, cat_key) for r in rows]
    if not diffs:
        print(f"  {name}: empty"); return
    n = len(diffs)
    print(f"  {name}: n={n}  difficulty span min={min(diffs):.3f} "
          f"med={sorted(diffs)[n//2]:.3f} max={max(diffs):.3f}  "
          f"first_5={[f'{d:.2f}' for d in diffs[:5]]} "
          f"last_5={[f'{d:.2f}' for d in diffs[-5:]]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-train", required=True,
                    help="DPO json (with 'axis' & 'cognitive_category').")
    ap.add_argument("--in-sft", required=True,
                    help="SFT-warmstart json (same fields).")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-sft", required=True)
    ap.add_argument("--baseline-results-dir", required=True,
                    help="Per-video judged JSON dir from a baseline benchmark "
                         "run; used to estimate per-(axis,cat) difficulty.")
    args = ap.parse_args()

    pair_acc, ax_acc = load_baseline_acc(args.baseline_results_dir)
    print(f"[curriculum] loaded {len(pair_acc)} (axis,cat) cells from "
          f"{args.baseline_results_dir}")
    print("  per-axis baseline acc:")
    for k, v in sorted(ax_acc.items()):
        print(f"    {k:10s} acc={100*v:5.1f}% (difficulty={1-v:.2f})")

    train = json.load(open(args.in_train))
    sft = json.load(open(args.in_sft))

    # DPO rows have 'axis' + 'cognitive_category' at top level.
    train_sorted = sort_one(train, pair_acc, ax_acc, "axis", "cognitive_category")
    # SFT rows from sof_dpo_make_sft_warmstart.py also keep 'axis' + 'cognitive_category'.
    sft_sorted = sort_one(sft, pair_acc, ax_acc, "axis", "cognitive_category")

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    json.dump(train_sorted, open(args.out_train, "w"), indent=2)
    json.dump(sft_sorted, open(args.out_sft, "w"), indent=2)
    print(f"\n[curriculum] wrote {len(train_sorted)} -> {args.out_train}")
    print(f"[curriculum] wrote {len(sft_sorted)} -> {args.out_sft}")

    print("\n=== sanity: easy-first ordering ===")
    report("DPO train", train_sorted, pair_acc, ax_acc,
           "axis", "cognitive_category")
    report("SFT train", sft_sorted, pair_acc, ax_acc,
           "axis", "cognitive_category")


if __name__ == "__main__":
    main()
