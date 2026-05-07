#!/usr/bin/env python3
"""
===============================================================================
Human Evaluation Tool — Side-by-Side QA Review
===============================================================================
Presents a random sample of predictions for manual scoring in the terminal.

For each sample the reviewer sees:
  • The question
  • The model's prediction
  • The ground-truth reference answer
  • The original MCQ choices + correct answer (if CGBench)

The reviewer rates each on a 1-5 scale:
  1 = Completely wrong / irrelevant
  2 = Partially relevant but mostly incorrect
  3 = Partially correct, captures some key points
  4 = Mostly correct with minor omissions or extra info
  5 = Fully correct and comprehensive

Special keys:
  s = skip (don't score, move to next)
  q = quit early and save what you have so far
  f = flag for later discussion

Results are saved to a JSON file for analysis.

Usage:
    # Default: 100 random samples
    python tools/human_eval.py \
        --predictions_path outputs/V-05_.../test_results/predictions.json

    # Custom sample size
    python tools/human_eval.py \
        --predictions_path predictions.json \
        --num_samples 50

    # Compare two models side-by-side
    python tools/human_eval.py \
        --predictions_path outputs/model_A/predictions.json \
        --compare_path outputs/model_B/predictions.json \
        --num_samples 100

    # Seed for reproducibility
    python tools/human_eval.py \
        --predictions_path predictions.json \
        --seed 42

    # Resume a previous session
    python tools/human_eval.py \
        --predictions_path predictions.json \
        --resume results/human_eval_20260415_1030.json
===============================================================================
"""

import argparse
import json
import os
import random
import sys
import textwrap
from datetime import datetime

# =========================================================================
# ANSI helpers
# =========================================================================

class C:
    """ANSI colour codes for terminal output."""
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    UNDERLINE = "\033[4m"


def wrap(text: str, width: int = 78, indent: str = "    ") -> str:
    """Word-wrap text with indent for display."""
    lines = textwrap.wrap(text, width=width - len(indent))
    return "\n".join(indent + l for l in lines) if lines else indent + "(empty)"


# =========================================================================
# Load CGBench lookup (optional enrichment)
# =========================================================================

_DEFAULT_CGBENCH = os.path.join(
    os.path.dirname(__file__), "..", "..", "cgbench_setup", "cgbench.json"
)


def load_cgbench_lookup(cgbench_path: str) -> dict:
    if not os.path.exists(cgbench_path):
        return {}
    try:
        with open(cgbench_path) as f:
            data = json.load(f)
        return {str(item["qid"]): item for item in data}
    except Exception:
        return {}


# =========================================================================
# Display a single sample for review
# =========================================================================

def display_sample(idx: int, total: int, sample: dict, cg_item: dict | None,
                   compare_pred: str | None = None):
    """Print a single sample for the reviewer."""
    print(f"\n{'═' * 78}")
    print(f"  {C.BOLD}Sample {idx + 1}/{total}{C.RESET}"
          f"  {C.DIM}[id: {sample.get('id', '?')}]{C.RESET}")
    print(f"{'═' * 78}")

    # Question (extract from conversations or metadata)
    question = sample.get("question", "")
    if not question:
        # Try to extract from conversations in the predictions file
        convs = sample.get("conversations", [])
        for t in convs:
            if t.get("from", t.get("role", "")) in ("human", "user"):
                question = t.get("value", t.get("content", ""))
                question = question.replace("<video>", "").replace("<image>", "").strip()
                break

    # If still no question, try to reconstruct from reference context
    if not question and cg_item:
        question = cg_item.get("question", "(question not available)")

    print(f"\n  {C.CYAN}{C.BOLD}Question:{C.RESET}")
    print(wrap(question))

    # Domain / sub-category
    meta = sample.get("metadata", {})
    domain = meta.get("domain", "")
    subcat = meta.get("sub_category", "")
    if domain or subcat:
        print(f"\n  {C.DIM}Domain: {domain}  |  Category: {subcat}{C.RESET}")

    # Model prediction
    print(f"\n  {C.YELLOW}{C.BOLD}Model Prediction:{C.RESET}")
    print(wrap(sample.get("prediction", "(no prediction)")))

    # Compare prediction (if comparing two models)
    if compare_pred is not None:
        print(f"\n  {C.MAGENTA}{C.BOLD}Model B Prediction:{C.RESET}")
        print(wrap(compare_pred))

    # Reference answer
    print(f"\n  {C.GREEN}{C.BOLD}Reference Answer:{C.RESET}")
    print(wrap(sample.get("reference", "(no reference)")))

    # Original MCQ info
    if cg_item:
        print(f"\n  {C.BLUE}{C.BOLD}Original MCQ Choices:{C.RESET}")
        choices = cg_item.get("choices", [])
        gold_key = cg_item.get("right_answer", "?").upper()
        gold_idx = ord(gold_key) - ord("A") if gold_key.isalpha() else -1
        for i, ch in enumerate(choices):
            letter = chr(ord("A") + i)
            marker = f" {C.GREEN}← correct{C.RESET}" if i == gold_idx else ""
            print(f"    {C.DIM}{letter}.{C.RESET} {ch}{marker}")

    # BLEU/ROUGE if available
    metrics = sample.get("metrics", {})
    if metrics:
        bleu4 = metrics.get("bleu_4", "?")
        rouge_l = metrics.get("rouge_l", "?")
        print(f"\n  {C.DIM}Auto-metrics: BLEU-4={bleu4}  ROUGE-L={rouge_l}{C.RESET}")

    print(f"\n{'─' * 78}")


def get_score(compare_mode: bool = False) -> dict:
    """
    Prompt the reviewer for a score.
    Returns dict with keys: score_a (int|None), score_b (int|None),
    action ('score'|'skip'|'quit'|'flag'), note (str).
    """
    score_label = "Score" if not compare_mode else "Score Model A"
    prompt_parts = [
        f"  {C.BOLD}{score_label} [1-5]{C.RESET}",
        f"{C.DIM}(s=skip, f=flag, q=quit){C.RESET}: ",
    ]
    prompt = ", ".join(prompt_parts)

    result = {"score_a": None, "score_b": None, "action": "score", "note": ""}

    while True:
        raw = input(prompt).strip().lower()
        if raw == "q":
            result["action"] = "quit"
            return result
        if raw == "s":
            result["action"] = "skip"
            return result
        if raw.startswith("f"):
            result["action"] = "flag"
            note = input(f"  {C.DIM}Optional note: {C.RESET}").strip()
            result["note"] = note
            # Still get a score for flagged items
            raw = input(f"  {C.BOLD}Score anyway [1-5] or s to skip: {C.RESET}").strip().lower()
            if raw == "s":
                return result
            try:
                score = int(raw)
                if 1 <= score <= 5:
                    result["score_a"] = score
                    break
            except ValueError:
                pass
            print(f"  {C.RED}Please enter 1-5 or s.{C.RESET}")
            continue

        try:
            score = int(raw)
            if 1 <= score <= 5:
                result["score_a"] = score
                break
        except ValueError:
            pass
        print(f"  {C.RED}Please enter 1-5, s, f, or q.{C.RESET}")

    # If comparing, get score for model B
    if compare_mode:
        while True:
            raw = input(f"  {C.BOLD}Score Model B [1-5]{C.RESET}: ").strip()
            try:
                score = int(raw)
                if 1 <= score <= 5:
                    result["score_b"] = score
                    break
            except ValueError:
                pass
            print(f"  {C.RED}Please enter 1-5.{C.RESET}")

    return result


# =========================================================================
# Results aggregation and saving
# =========================================================================

def compute_summary(results: list[dict], compare_mode: bool = False) -> dict:
    """Compute summary statistics from human eval results."""
    scored = [r for r in results if r.get("score_a") is not None]
    flagged = [r for r in results if r.get("action") == "flag"]
    skipped = [r for r in results if r.get("action") == "skip"]

    summary = {
        "total_reviewed": len(results),
        "scored": len(scored),
        "skipped": len(skipped),
        "flagged": len(flagged),
    }

    if scored:
        scores_a = [r["score_a"] for r in scored]
        summary["model_a"] = {
            "mean_score": round(sum(scores_a) / len(scores_a), 3),
            "score_distribution": {str(i): scores_a.count(i) for i in range(1, 6)},
            "pct_correct": round(100 * sum(1 for s in scores_a if s >= 4) / len(scores_a), 1),
            "pct_perfect": round(100 * sum(1 for s in scores_a if s == 5) / len(scores_a), 1),
        }

    if compare_mode:
        scored_b = [r for r in scored if r.get("score_b") is not None]
        if scored_b:
            scores_b = [r["score_b"] for r in scored_b]
            summary["model_b"] = {
                "mean_score": round(sum(scores_b) / len(scores_b), 3),
                "score_distribution": {str(i): scores_b.count(i) for i in range(1, 6)},
                "pct_correct": round(100 * sum(1 for s in scores_b if s >= 4) / len(scores_b), 1),
                "pct_perfect": round(100 * sum(1 for s in scores_b if s == 5) / len(scores_b), 1),
            }
            # Win/tie/loss
            a_wins = sum(1 for r in scored_b if r["score_a"] > r["score_b"])
            b_wins = sum(1 for r in scored_b if r["score_b"] > r["score_a"])
            ties = sum(1 for r in scored_b if r["score_a"] == r["score_b"])
            summary["comparison"] = {
                "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
                "a_win_rate": round(100 * a_wins / len(scored_b), 1) if scored_b else 0,
                "b_win_rate": round(100 * b_wins / len(scored_b), 1) if scored_b else 0,
            }

    return summary


def print_final_summary(summary: dict, compare_mode: bool = False):
    """Print a nice summary at the end."""
    print(f"\n\n{'═' * 70}")
    print(f"  {C.BOLD}HUMAN EVALUATION SUMMARY{C.RESET}")
    print(f"{'═' * 70}")
    print(f"  Reviewed: {summary['total_reviewed']}  |  "
          f"Scored: {summary['scored']}  |  "
          f"Skipped: {summary['skipped']}  |  "
          f"Flagged: {summary['flagged']}")

    if "model_a" in summary:
        ma = summary["model_a"]
        label = "Model A" if compare_mode else "Model"
        print(f"\n  {C.BOLD}{label}:{C.RESET}")
        print(f"    Mean score:  {ma['mean_score']:.2f} / 5.0")
        print(f"    ≥4 (correct): {ma['pct_correct']}%")
        print(f"    =5 (perfect):  {ma['pct_perfect']}%")
        print(f"    Distribution: {ma['score_distribution']}")

    if "model_b" in summary:
        mb = summary["model_b"]
        print(f"\n  {C.BOLD}Model B:{C.RESET}")
        print(f"    Mean score:  {mb['mean_score']:.2f} / 5.0")
        print(f"    ≥4 (correct): {mb['pct_correct']}%")
        print(f"    =5 (perfect):  {mb['pct_perfect']}%")
        print(f"    Distribution: {mb['score_distribution']}")

    if "comparison" in summary:
        comp = summary["comparison"]
        print(f"\n  {C.BOLD}Head-to-Head:{C.RESET}")
        print(f"    A wins: {comp['a_wins']} ({comp['a_win_rate']}%)")
        print(f"    B wins: {comp['b_wins']} ({comp['b_win_rate']}%)")
        print(f"    Ties:   {comp['ties']}")

    print(f"{'═' * 70}\n")


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Human evaluation of QA predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--predictions_path", type=str, required=True,
                   help="Path to predictions.json file (Model A).")
    p.add_argument("--compare_path", type=str, default=None,
                   help="Path to second predictions.json for side-by-side comparison (Model B).")
    p.add_argument("--cgbench_path", type=str, default=_DEFAULT_CGBENCH,
                   help="Path to original cgbench.json for MCQ context.")
    p.add_argument("--num_samples", type=int, default=100,
                   help="Number of samples to review (default: 100).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sample selection.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for results. Default: alongside predictions file.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a previous human_eval results JSON to resume from.")
    p.add_argument("--stratified", action="store_true",
                   help="Stratify sampling across domains (proportional).")
    return p.parse_args()


def main():
    args = parse_args()

    # Load predictions
    with open(args.predictions_path) as f:
        predictions_a = json.load(f)
    print(f"Loaded {len(predictions_a)} predictions from {args.predictions_path}")

    # Load comparison predictions
    predictions_b_by_id = {}
    compare_mode = args.compare_path is not None
    if compare_mode:
        with open(args.compare_path) as f:
            predictions_b = json.load(f)
        predictions_b_by_id = {p["id"]: p for p in predictions_b}
        print(f"Loaded {len(predictions_b)} comparison predictions from {args.compare_path}")

    # Load CGBench for MCQ enrichment
    cg_lookup = load_cgbench_lookup(args.cgbench_path)
    if cg_lookup:
        print(f"Loaded {len(cg_lookup)} CGBench MCQ items for context.")

    # Enrich predictions with question text from test data
    # (predictions files may not have the question)
    test_data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "final_sft_data", "test.json"
    )
    test_by_id = {}
    if os.path.exists(test_data_path):
        with open(test_data_path) as f:
            test_data = json.load(f)
        for t in test_data:
            tid = t.get("id", "")
            question = ""
            for conv in t.get("conversations", []):
                if conv.get("from", conv.get("role", "")) in ("human", "user"):
                    question = conv.get("value", "").replace("<video>", "").replace("<image>", "").strip()
                    break
            test_by_id[tid] = {"question": question, "conversations": t.get("conversations", [])}

    # Add question to predictions
    for p in predictions_a:
        pid = p.get("id", "")
        if pid in test_by_id:
            p["question"] = test_by_id[pid]["question"]
            p["conversations"] = test_by_id[pid]["conversations"]

    # Sample selection
    random.seed(args.seed)
    n = min(args.num_samples, len(predictions_a))

    if args.stratified and cg_lookup:
        # Group by domain, sample proportionally
        by_domain = {}
        for p in predictions_a:
            orig_id = str(p.get("metadata", {}).get("original_id", ""))
            cg = cg_lookup.get(orig_id, {})
            domain = cg.get("domain", p.get("metadata", {}).get("domain", "unknown"))
            by_domain.setdefault(domain, []).append(p)

        samples = []
        total_available = sum(len(v) for v in by_domain.values())
        for domain, items in by_domain.items():
            k = max(1, round(n * len(items) / total_available))
            samples.extend(random.sample(items, min(k, len(items))))
        # Trim or pad to exactly n
        random.shuffle(samples)
        samples = samples[:n]
    else:
        samples = random.sample(predictions_a, n)

    # Resume handling
    already_done = set()
    previous_results = []
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            resumed = json.load(f)
        previous_results = resumed.get("results", [])
        already_done = {r["id"] for r in previous_results}
        print(f"Resumed {len(previous_results)} previous results.")

    # Output directory
    output_dir = args.output_dir or os.path.dirname(args.predictions_path)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"human_eval_{timestamp}.json")

    # ── Main review loop ─────────────────────────────────────────────────
    print(f"\n{C.BOLD}Starting human evaluation: {n} samples{C.RESET}")
    print(f"  Score 1-5 | s=skip | f=flag | q=quit\n")

    results = list(previous_results)
    reviewed_count = len(previous_results)

    for idx, sample in enumerate(samples):
        sid = sample.get("id", "")
        if sid in already_done:
            continue

        # Get CGBench item
        orig_id = str(sample.get("metadata", {}).get("original_id", ""))
        cg_item = cg_lookup.get(orig_id)

        # Get comparison prediction
        compare_pred = None
        if compare_mode and sid in predictions_b_by_id:
            compare_pred = predictions_b_by_id[sid].get("prediction", "")

        # Display
        display_sample(reviewed_count, n, sample, cg_item, compare_pred)

        # Get score
        score_result = get_score(compare_mode)

        if score_result["action"] == "quit":
            print(f"\n{C.YELLOW}Quitting early. Saving {len(results)} results...{C.RESET}")
            break

        entry = {
            "id": sid,
            "score_a": score_result["score_a"],
            "action": score_result["action"],
            "note": score_result.get("note", ""),
            "prediction": sample.get("prediction", ""),
            "reference": sample.get("reference", ""),
            "domain": sample.get("metadata", {}).get("domain", ""),
            "sub_category": sample.get("metadata", {}).get("sub_category", ""),
        }
        if compare_mode:
            entry["score_b"] = score_result["score_b"]
            entry["compare_prediction"] = compare_pred

        results.append(entry)
        reviewed_count += 1

        # Auto-save every 10 samples
        if reviewed_count % 10 == 0:
            _save_results(output_path, results, args, compare_mode)
            print(f"  {C.DIM}[auto-saved {reviewed_count} results]{C.RESET}")

    # Final save
    _save_results(output_path, results, args, compare_mode)

    # Print summary
    summary = compute_summary(results, compare_mode)
    print_final_summary(summary, compare_mode)
    print(f"  Results saved to: {C.UNDERLINE}{output_path}{C.RESET}\n")


def _save_results(path: str, results: list, args, compare_mode: bool):
    """Save results + summary to JSON."""
    summary = compute_summary(results, compare_mode)
    output = {
        "predictions_path": args.predictions_path,
        "compare_path": args.compare_path,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
