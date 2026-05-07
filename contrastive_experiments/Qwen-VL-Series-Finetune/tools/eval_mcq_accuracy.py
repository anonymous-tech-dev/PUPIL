#!/usr/bin/env python3
"""
===============================================================================
MCQ-Grounded Accuracy Evaluator
===============================================================================
Since the test set was reverse-engineered from CGBench MCQ → open-ended QA,
we can recover the original choices and measure *accuracy* — not just BLEU/ROUGE.

For every prediction, we score it against each original MCQ choice using three
signals combined into a single score:

  score(pred, choice) = 2·substring + 1.5·word_overlap + sequence_ratio

    • substring   — 1 if the normalised choice appears verbatim inside the
                    normalised prediction, else 0.  Captures exact matches
                    (e.g. "278 yen" in "The price is 278 yen.").
    • word_overlap — |words(pred) ∩ words(choice)| / |words(choice)|.
                    Captures paraphrase / reordering.
    • sequence_ratio — SequenceMatcher ratio (longest common subsequence).
                    Soft fallback for partial overlap.

The prediction is assigned to whichever choice scores highest.
If that choice is the gold answer → correct.

Reports:
  • Overall MCQ accuracy
  • Per-domain, per-sub_category accuracy
  • Confidence margin histogram (gap between top-1 and top-2 scores)
  • Low-confidence samples for human review
  • Side-by-side with BLEU/ROUGE from the predictions file

Usage examples:
    # Evaluate a single predictions file
    python tools/eval_mcq_accuracy.py \
        --predictions_path outputs/V-05_.../test_results/predictions.json

    # Evaluate ALL predictions files under an outputs root
    python tools/eval_mcq_accuracy.py \
        --outputs_root /workspace/Pupil/contrastive_experiments/outputs/

    python tools/eval_mcq_accuracy.py \
        --outputs_root outputs/


    # Custom CGBench path
    python tools/eval_mcq_accuracy.py \
        --predictions_path predictions.json \
        --cgbench_path /path/to/cgbench.json

    # Save detailed per-sample results
    python tools/eval_mcq_accuracy.py \
        --predictions_path predictions.json \
        --save_details
===============================================================================
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


# =========================================================================
# Text normalisation
# =========================================================================

def normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================================
# Choice scoring
# =========================================================================

def score_choice(pred_norm: str, pred_words: set, choice_norm: str, choice_words: set) -> float:
    """
    Combined score for how well *pred* matches a single *choice*.
    Higher = better match.
    """
    # Signal 1 — exact substring
    substr = 1.0 if choice_norm in pred_norm else 0.0

    # Signal 2 — word-level recall (what fraction of choice words appear in pred)
    if choice_words:
        overlap = len(pred_words & choice_words) / len(choice_words)
    else:
        overlap = 0.0

    # Signal 3 — SequenceMatcher ratio (soft LCS)
    seq = SequenceMatcher(None, pred_norm, choice_norm).ratio()

    return 2.0 * substr + 1.5 * overlap + seq


def match_prediction_to_choices(prediction: str, choices: list[str]):
    """
    Return (best_idx, best_score, margin, all_scores).
    *margin* = best_score − second_best_score (higher = more confident).
    """
    pred_norm = normalise(prediction)
    pred_words = set(pred_norm.split())

    scores = []
    for i, c in enumerate(choices):
        c_norm = normalise(c)
        c_words = set(c_norm.split())
        s = score_choice(pred_norm, pred_words, c_norm, c_words)
        scores.append((s, i))

    scores.sort(reverse=True)
    best_score, best_idx = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    margin = best_score - second_score

    all_scores = {choices[idx]: sc for sc, idx in sorted(scores, key=lambda x: x[1])}
    return best_idx, best_score, margin, all_scores


# =========================================================================
# Load CGBench lookup
# =========================================================================

_DEFAULT_CGBENCH = os.path.join(
    os.path.dirname(__file__), "..", "..", "cgbench_setup", "cgbench.json"
)


def load_cgbench_lookup(cgbench_path: str) -> dict:
    """Return {str(qid): item, ...} from the original CGBench MCQ file."""
    with open(cgbench_path) as f:
        data = json.load(f)
    return {str(item["qid"]): item for item in data}


# =========================================================================
# Evaluate one predictions file
# =========================================================================

def evaluate_predictions(predictions: list[dict], cg_lookup: dict, verbose: bool = False):
    """
    Returns a dict with:
      overall_accuracy, per_domain, per_subcategory, per_sample (list),
      confidence_stats, low_confidence_samples
    """
    results = []
    correct = 0
    total = 0

    per_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    per_subcat = defaultdict(lambda: {"correct": 0, "total": 0})

    margins = []

    for p in predictions:
        orig_id = str(p.get("metadata", {}).get("original_id", ""))
        cg = cg_lookup.get(orig_id)
        if cg is None:
            # Not a CGBench sample — skip
            continue

        total += 1
        choices = cg["choices"]
        gold_key = cg["right_answer"].upper()
        gold_idx = ord(gold_key) - ord("A")
        gold_answer = cg["answer"]

        pred_idx, top_score, margin, all_scores = match_prediction_to_choices(
            p["prediction"], choices
        )
        is_correct = pred_idx == gold_idx
        if is_correct:
            correct += 1

        domain = p.get("metadata", {}).get("domain", cg.get("domain", "unknown"))
        subcat = p.get("metadata", {}).get("sub_category", cg.get("sub_category", "unknown"))

        per_domain[domain]["total"] += 1
        per_subcat[subcat]["total"] += 1
        if is_correct:
            per_domain[domain]["correct"] += 1
            per_subcat[subcat]["correct"] += 1

        margins.append(margin)

        sample_result = {
            "id": p.get("id", ""),
            "prediction": p["prediction"],
            "reference": p.get("reference", ""),
            "gold_answer_mcq": gold_answer,
            "matched_choice": choices[pred_idx],
            "matched_idx": pred_idx,
            "gold_idx": gold_idx,
            "is_correct": is_correct,
            "top_score": round(top_score, 4),
            "margin": round(margin, 4),
            "domain": domain,
            "sub_category": subcat,
            "bleu_4": p.get("metrics", {}).get("bleu_4"),
            "rouge_l": p.get("metrics", {}).get("rouge_l"),
        }
        results.append(sample_result)

    # Aggregate
    accuracy = correct / total if total > 0 else 0.0

    domain_acc = {}
    for d, v in sorted(per_domain.items()):
        domain_acc[d] = {
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
            "correct": v["correct"],
            "total": v["total"],
        }

    subcat_acc = {}
    for s, v in sorted(per_subcat.items()):
        subcat_acc[s] = {
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
            "correct": v["correct"],
            "total": v["total"],
        }

    # Confidence analysis
    if margins:
        avg_margin = sum(margins) / len(margins)
        low_conf_threshold = avg_margin * 0.3  # bottom 30% relative
        low_conf_samples = [r for r in results if r["margin"] < low_conf_threshold]
    else:
        avg_margin = 0.0
        low_conf_samples = []

    # Correlation with BLEU/ROUGE
    bleu_correct = [r["bleu_4"] for r in results if r["is_correct"] and r["bleu_4"] is not None]
    bleu_wrong = [r["bleu_4"] for r in results if not r["is_correct"] and r["bleu_4"] is not None]
    rouge_correct = [r["rouge_l"] for r in results if r["is_correct"] and r["rouge_l"] is not None]
    rouge_wrong = [r["rouge_l"] for r in results if not r["is_correct"] and r["rouge_l"] is not None]

    correlation = {}
    if bleu_correct:
        correlation["avg_bleu4_when_correct"] = sum(bleu_correct) / len(bleu_correct)
    if bleu_wrong:
        correlation["avg_bleu4_when_wrong"] = sum(bleu_wrong) / len(bleu_wrong)
    if rouge_correct:
        correlation["avg_rougeL_when_correct"] = sum(rouge_correct) / len(rouge_correct)
    if rouge_wrong:
        correlation["avg_rougeL_when_wrong"] = sum(rouge_wrong) / len(rouge_wrong)

    return {
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_domain": domain_acc,
        "per_sub_category": subcat_acc,
        "confidence": {
            "avg_margin": round(avg_margin, 4),
            "num_low_confidence": len(low_conf_samples),
            "low_confidence_threshold": round(low_conf_threshold, 4) if margins else 0.0,
        },
        "bleu_rouge_correlation": correlation,
        "per_sample": results,
        "low_confidence_samples": low_conf_samples[:50],  # cap for readability
    }


# =========================================================================
# Pretty printing
# =========================================================================

def print_summary(result: dict, label: str = ""):
    """Print a human-readable summary."""
    if label:
        print(f"\n{'═' * 70}")
        print(f"  {label}")
        print(f"{'═' * 70}")

    print(f"\n  MCQ Accuracy: {result['correct']}/{result['total']}"
          f"  ({100 * result['overall_accuracy']:.1f}%)")

    # Per-domain
    if result["per_domain"]:
        print(f"\n  {'Domain':<35} {'Acc':>7} {'n':>5}")
        print(f"  {'─' * 50}")
        for d, v in sorted(result["per_domain"].items(), key=lambda x: -x[1]["accuracy"]):
            print(f"  {d:<35} {100*v['accuracy']:>6.1f}% {v['total']:>5}")

    # Per sub_category
    if result["per_sub_category"]:
        print(f"\n  {'Sub-Category':<35} {'Acc':>7} {'n':>5}")
        print(f"  {'─' * 50}")
        for s, v in sorted(result["per_sub_category"].items(), key=lambda x: -x[1]["accuracy"]):
            print(f"  {s:<35} {100*v['accuracy']:>6.1f}% {v['total']:>5}")

    # Confidence
    conf = result["confidence"]
    print(f"\n  Confidence: avg margin = {conf['avg_margin']:.3f}, "
          f"low-confidence samples = {conf['num_low_confidence']}")

    # BLEU/ROUGE correlation
    corr = result.get("bleu_rouge_correlation", {})
    if corr:
        print(f"\n  BLEU-4 / ROUGE-L when correct vs wrong:")
        bc = corr.get("avg_bleu4_when_correct", 0)
        bw = corr.get("avg_bleu4_when_wrong", 0)
        rc = corr.get("avg_rougeL_when_correct", 0)
        rw = corr.get("avg_rougeL_when_wrong", 0)
        print(f"    BLEU-4:  correct={bc:.4f}  wrong={bw:.4f}  (Δ={bc-bw:+.4f})")
        print(f"    ROUGE-L: correct={rc:.4f}  wrong={rw:.4f}  (Δ={rc-rw:+.4f})")


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate QA predictions using original CGBench MCQ choices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--predictions_path", type=str,
                   help="Path to a single predictions.json file.")
    g.add_argument("--outputs_root", type=str,
                   help="Root dir to scan for **/predictions.json and evaluate all.")

    p.add_argument("--cgbench_path", type=str, default=_DEFAULT_CGBENCH,
                   help="Path to original cgbench.json with MCQ data.")
    p.add_argument("--save_details", action="store_true",
                   help="Save per-sample details alongside the predictions file.")
    p.add_argument("--quiet", action="store_true",
                   help="Only print the final leaderboard (for --outputs_root).")
    return p.parse_args()


def main():
    args = parse_args()

    # Load CGBench lookup
    cgbench_path = os.path.normpath(args.cgbench_path)
    if not os.path.exists(cgbench_path):
        print(f"ERROR: CGBench file not found: {cgbench_path}")
        sys.exit(1)
    cg_lookup = load_cgbench_lookup(cgbench_path)
    print(f"Loaded {len(cg_lookup)} CGBench MCQ items from {cgbench_path}")

    # Collect prediction files
    if args.predictions_path:
        pred_files = [args.predictions_path]
    else:
        pred_files = sorted(
            glob.glob(os.path.join(args.outputs_root, "**", "predictions.json"), recursive=True)
        )
        if not pred_files:
            print(f"No predictions.json found under {args.outputs_root}")
            sys.exit(1)
        print(f"Found {len(pred_files)} prediction files.\n")

    # Evaluate each
    leaderboard = []
    for pf in pred_files:
        with open(pf) as f:
            predictions = json.load(f)

        result = evaluate_predictions(predictions, cg_lookup)

        # Label
        if args.outputs_root:
            label = os.path.relpath(pf, args.outputs_root)
        else:
            label = pf

        if not args.quiet:
            print_summary(result, label)

        leaderboard.append((label, result["overall_accuracy"], result["correct"], result["total"]))

        # Save details
        if args.save_details:
            detail_path = os.path.join(os.path.dirname(pf), "mcq_accuracy.json")
            # Strip per_sample for the summary file, save separately
            summary = {k: v for k, v in result.items() if k != "per_sample"}
            with open(detail_path, "w") as f:
                json.dump(summary, f, indent=2)

            detail_samples_path = os.path.join(os.path.dirname(pf), "mcq_accuracy_details.json")
            with open(detail_samples_path, "w") as f:
                json.dump(result["per_sample"], f, indent=2, ensure_ascii=False)

            print(f"\n  Saved: {detail_path}")
            print(f"  Saved: {detail_samples_path}")

    # Leaderboard
    if len(leaderboard) > 1:
        print(f"\n\n{'═' * 80}")
        print("  LEADERBOARD (sorted by MCQ accuracy)")
        print(f"{'═' * 80}")
        print(f"  {'#':<3} {'Experiment':<62} {'Acc':>7} {'n':>5}")
        print(f"  {'─' * 78}")
        for rank, (label, acc, c, t) in enumerate(
            sorted(leaderboard, key=lambda x: -x[1]), 1
        ):
            marker = " 🏆" if rank == 1 else ""
            print(f"  {rank:<3} {label:<62} {100*acc:>6.1f}% {t:>5}{marker}")

    print()


if __name__ == "__main__":
    main()
