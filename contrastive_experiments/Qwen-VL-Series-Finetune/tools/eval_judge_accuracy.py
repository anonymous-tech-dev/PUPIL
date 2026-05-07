#!/usr/bin/env python3
"""
===============================================================================
GPT-5 LLM-as-Judge Accuracy Analyser
===============================================================================
Analyses gpt5_judge.json files produced by the GPT-5 MCQ judge pipeline.
Each JSON already contains per-sample verdicts (is_correct, chosen_letter,
confidence, reason), so no re-scoring is needed — we simply aggregate.

Reports:
  • Overall judge accuracy
  • Per-domain, per-sub_category accuracy
  • Per-confidence-level accuracy  (low / medium / high)
  • Error count & error samples
  • Leaderboard across all experiments (when using --outputs_root)

Usage examples:
    # Single file
    python tools/eval_judge_accuracy.py \
        --judge_path outputs/V-02_.../test_results_full_video/gpt5_judge.json

    # All judge files under an outputs root
    python tools/eval_judge_accuracy.py \
        --outputs_root /workspace/Pupil/contrastive_experiments/outputs/

    # Quiet mode — leaderboard only
    python tools/eval_judge_accuracy.py \
        --outputs_root outputs/ --quiet

    # Save per-sample detail JSONs next to each judge file
    python tools/eval_judge_accuracy.py \
        --outputs_root outputs/ --save_details
===============================================================================
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


# =========================================================================
# Analyse one judge JSON
# =========================================================================

def analyse_judge_file(data: dict):
    """
    Accepts the parsed contents of a gpt5_judge.json and returns an
    aggregated result dict.
    """
    samples = data.get("samples", [])

    correct = 0
    total = 0
    errors = 0

    per_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    per_subcat = defaultdict(lambda: {"correct": 0, "total": 0})
    per_confidence = defaultdict(lambda: {"correct": 0, "total": 0})

    error_samples = []
    per_sample = []

    for s in samples:
        total += 1
        is_correct = s.get("is_correct", False)
        domain = s.get("domain", "unknown")
        subcat = s.get("sub_category", "unknown")
        confidence = s.get("confidence", "unknown")

        if s.get("error"):
            errors += 1
            error_samples.append(s)

        if is_correct:
            correct += 1

        per_domain[domain]["total"] += 1
        per_subcat[subcat]["total"] += 1
        per_confidence[confidence]["total"] += 1
        if is_correct:
            per_domain[domain]["correct"] += 1
            per_subcat[subcat]["correct"] += 1
            per_confidence[confidence]["correct"] += 1

        per_sample.append({
            "id": s.get("id", ""),
            "question": s.get("question", ""),
            "prediction": s.get("prediction", ""),
            "gold_letter": s.get("gold_letter", ""),
            "gold_answer_text": s.get("gold_answer_text", ""),
            "chosen_letter": s.get("chosen_letter", ""),
            "confidence": confidence,
            "is_correct": is_correct,
            "domain": domain,
            "sub_category": subcat,
            "reason": s.get("reason", ""),
            "error": s.get("error"),
        })

    accuracy = correct / total if total > 0 else 0.0

    def _acc_dict(dd):
        return {
            k: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in sorted(dd.items())
        }

    return {
        "model": data.get("model", "unknown"),
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "errors": errors,
        "per_domain": _acc_dict(per_domain),
        "per_sub_category": _acc_dict(per_subcat),
        "per_confidence": _acc_dict(per_confidence),
        "error_samples": error_samples[:30],
        "per_sample": per_sample,
    }


# =========================================================================
# Pretty printing
# =========================================================================

def print_summary(result: dict, label: str = ""):
    if label:
        print(f"\n{'═' * 70}")
        print(f"  {label}")
        print(f"{'═' * 70}")

    print(f"\n  Judge model : {result['model']}")
    print(f"  Accuracy    : {result['correct']}/{result['total']}"
          f"  ({100 * result['overall_accuracy']:.1f}%)")
    if result["errors"]:
        print(f"  Errors      : {result['errors']}")

    # Per-confidence
    if result["per_confidence"]:
        print(f"\n  {'Confidence':<15} {'Acc':>7} {'n':>5}")
        print(f"  {'─' * 30}")
        for c, v in sorted(result["per_confidence"].items(),
                           key=lambda x: -x[1]["accuracy"]):
            print(f"  {c:<15} {100*v['accuracy']:>6.1f}% {v['total']:>5}")

    # Per-domain
    if result["per_domain"]:
        print(f"\n  {'Domain':<35} {'Acc':>7} {'n':>5}")
        print(f"  {'─' * 50}")
        for d, v in sorted(result["per_domain"].items(),
                           key=lambda x: -x[1]["accuracy"]):
            print(f"  {d:<35} {100*v['accuracy']:>6.1f}% {v['total']:>5}")

    # Per sub_category
    if result["per_sub_category"]:
        print(f"\n  {'Sub-Category':<35} {'Acc':>7} {'n':>5}")
        print(f"  {'─' * 50}")
        for s, v in sorted(result["per_sub_category"].items(),
                           key=lambda x: -x[1]["accuracy"]):
            print(f"  {s:<35} {100*v['accuracy']:>6.1f}% {v['total']:>5}")


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse GPT-5 LLM-as-Judge results and produce a leaderboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--judge_path", type=str,
                   help="Path to a single gpt5_judge.json file.")
    g.add_argument("--outputs_root", type=str,
                   help="Root dir to scan for **/gpt5_judge.json and evaluate all.")

    p.add_argument("--save_details", action="store_true",
                   help="Save per-sample detail JSON alongside each judge file.")
    p.add_argument("--quiet", action="store_true",
                   help="Only print the final leaderboard (for --outputs_root).")
    return p.parse_args()


def main():
    args = parse_args()

    # Collect judge files
    if args.judge_path:
        judge_files = [args.judge_path]
    else:
        judge_files = sorted(
            glob.glob(os.path.join(args.outputs_root, "**", "gpt5_judge.json"),
                       recursive=True)
        )
        if not judge_files:
            print(f"No gpt5_judge.json found under {args.outputs_root}")
            sys.exit(1)
        print(f"Found {len(judge_files)} judge file(s).\n")

    # Evaluate each
    leaderboard = []
    for jf in judge_files:
        with open(jf) as f:
            data = json.load(f)

        result = analyse_judge_file(data)

        # Label
        if args.outputs_root:
            label = os.path.relpath(jf, args.outputs_root)
        else:
            label = jf

        if not args.quiet:
            print_summary(result, label)

        leaderboard.append((
            label,
            result["overall_accuracy"],
            result["correct"],
            result["total"],
            result["errors"],
        ))

        # Save details
        if args.save_details:
            summary = {k: v for k, v in result.items()
                       if k not in ("per_sample", "error_samples")}
            summary_path = os.path.join(os.path.dirname(jf),
                                        "judge_accuracy_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            detail_path = os.path.join(os.path.dirname(jf),
                                       "judge_accuracy_details.json")
            with open(detail_path, "w") as f:
                json.dump(result["per_sample"], f, indent=2, ensure_ascii=False)

            print(f"\n  Saved: {summary_path}")
            print(f"  Saved: {detail_path}")

    # Leaderboard
    if len(leaderboard) > 1:
        print(f"\n\n{'═' * 90}")
        print("  LEADERBOARD — GPT-5 Judge Accuracy (sorted)")
        print(f"{'═' * 90}")
        print(f"  {'#':<3} {'Experiment':<62} {'Acc':>7} {'n':>5} {'Err':>5}")
        print(f"  {'─' * 85}")
        for rank, (label, acc, c, t, errs) in enumerate(
            sorted(leaderboard, key=lambda x: -x[1]), 1
        ):
            marker = " 🏆" if rank == 1 else ""
            err_str = str(errs) if errs else "-"
            print(f"  {rank:<3} {label:<62} {100*acc:>6.1f}% {t:>5} {err_str:>5}{marker}")

    print()


if __name__ == "__main__":
    main()
