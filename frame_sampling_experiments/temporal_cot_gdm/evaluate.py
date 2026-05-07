"""
evaluate.py — Evaluation Script for TCoT Results.

Run AFTER main.py has produced results.

Usage:
  python evaluate.py

Output:
  - Overall accuracy
  - Per question-type breakdown (LVBench)
  - Percentage of frames selected (vs total)
  - Comparison vs baseline (if baseline results present)

Hot-resume friendly: reads from the JSONL results file produced by main.py.
"""

import json
import os
import sys
from collections import defaultdict
from typing import List, Dict, Any

import config
from utils.results_io import load_all_results


def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall accuracy and per-question-type breakdown."""
    total   = 0
    correct = 0
    per_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        pred = r.get("predicted_letter", "").upper().strip()
        gt   = r.get("ground_truth", "").upper().strip()
        if not gt:
            continue  # skip items without ground truth

        total += 1
        is_correct = pred == gt
        if is_correct:
            correct += 1

        # Per question-type (LVBench)
        for qt in r.get("question_type", []):
            per_type[qt]["total"] += 1
            if is_correct:
                per_type[qt]["correct"] += 1

    accuracy = 100.0 * correct / total if total else 0.0

    return {
        "total"    : total,
        "correct"  : correct,
        "accuracy" : accuracy,
        "per_type" : {
            qt: {
                "total"   : v["total"],
                "correct" : v["correct"],
                "accuracy": 100.0 * v["correct"] / v["total"] if v["total"] else 0.0,
            }
            for qt, v in per_type.items()
        },
    }


def compute_selection_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute frame selection statistics."""
    if not results:
        return {}

    pcts = [r.get("pct_selected", 0.0) for r in results
            if r.get("total_frames", 0) > 0]
    num_sel = [r.get("num_selected", 0) for r in results]
    num_ctx = [r.get("num_context", 0) for r in results]

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    return {
        "mean_pct_selected"  : mean(pcts),
        "mean_num_selected"  : mean(num_sel),
        "mean_num_context"   : mean(num_ctx),
    }


def print_report(results: List[Dict[str, Any]]):
    print("=" * 65)
    print(f"  TCoT Evaluation Report")
    print(f"  Dataset : {config.DATASET}")
    print(f"  Model   : {config.MODEL}")
    print(f"  Variant : {config.TCOT_VARIANT}")
    print("=" * 65)

    if not results:
        print("  No results found. Run main.py first.")
        return

    # ── Accuracy ──────────────────────────────────────────────────────────
    acc_data = compute_accuracy(results)
    print(f"\n  Overall Accuracy : {acc_data['accuracy']:.2f}%"
          f"  ({acc_data['correct']}/{acc_data['total']})")

    # ── Per question type ─────────────────────────────────────────────────
    if acc_data["per_type"]:
        print("\n  Per Question Type:")
        print(f"  {'Type':<35} {'Acc':>6}  {'N':>5}")
        print("  " + "-" * 50)
        for qt, v in sorted(acc_data["per_type"].items(),
                             key=lambda x: -x[1]["accuracy"]):
            print(f"  {qt:<35} {v['accuracy']:>6.1f}%  {v['total']:>5}")

    # ── Frame selection stats ─────────────────────────────────────────────
    sel = compute_selection_stats(results)
    if sel:
        print("\n  Frame Selection Stats:")
        print(f"  Mean % frames selected : {sel['mean_pct_selected']:.1f}%")
        print(f"  Mean # frames selected : {sel['mean_num_selected']:.1f}")
        print(f"  Mean # context frames  : {sel['mean_num_context']:.1f}")

    # ── Error analysis ────────────────────────────────────────────────────
    no_pred = sum(1 for r in results if not r.get("predicted_letter"))
    if no_pred:
        print(f"\n  WARNING: {no_pred} items had no extractable prediction.")

    print("\n" + "=" * 65)


def print_comparison(tcot_results, baseline_results, native_results):
    """Print a three-way comparison: native → uniform → TCoT."""
    has_baseline = bool(baseline_results)
    has_native   = bool(native_results)

    if not has_baseline and not has_native:
        return

    tcot_acc    = compute_accuracy(tcot_results)
    base_acc    = compute_accuracy(baseline_results) if has_baseline else None
    native_acc  = compute_accuracy(native_results)   if has_native   else None

    print("\n" + "=" * 75)
    print("  Full Comparison: Native Baseline → Uniform Baseline → TCoT")
    print("=" * 75)
    print(f"  {'Method':<28} {'Accuracy':>10} {'N':>6} {'Δ vs Native':>12} {'Δ vs Uniform':>13}")
    print("  " + "-" * 71)

    def _row(label, acc_data, delta_native=None, delta_uniform=None):
        acc = acc_data["accuracy"]
        n   = acc_data["total"]
        dn  = f"{'+' if delta_native  >= 0 else ''}{delta_native:.2f}pp"  if delta_native  is not None else "    —"
        du  = f"{'+' if delta_uniform >= 0 else ''}{delta_uniform:.2f}pp" if delta_uniform is not None else "    —"
        print(f"  {label:<28} {acc:>9.2f}% {n:>6} {dn:>12} {du:>13}")

    native_a  = native_acc["accuracy"]  if native_acc  else None
    base_a    = base_acc["accuracy"]    if base_acc    else None
    tcot_a    = tcot_acc["accuracy"]

    if has_native:
        _row("Native (raw video → Qwen)", native_acc)
    if has_baseline:
        _row(f"Uniform k={config.CONTEXT_BUDGET_FRAMES} frames", base_acc,
             delta_native  = base_a - native_a if native_a is not None else None,
             delta_uniform = None)
    _row(f"TCoT ({config.TCOT_VARIANT})", tcot_acc,
         delta_native  = tcot_a - native_a if native_a is not None else None,
         delta_uniform = tcot_a - base_a   if base_a   is not None else None)

    # Per question type if available (LVBench)
    all_types = set(tcot_acc["per_type"].keys())
    if base_acc:
        all_types |= set(base_acc["per_type"].keys())
    if native_acc:
        all_types |= set(native_acc["per_type"].keys())

    if all_types:
        print(f"\n  Per Question Type (Δ = TCoT − {'Native' if has_native else 'Uniform'}):")
        ref_acc = native_acc if has_native else base_acc
        print(f"  {'Type':<32} {'Ref':>8} {'TCoT':>8} {'Delta':>8}")
        print("  " + "-" * 58)
        for qt in sorted(all_types):
            ref = ref_acc["per_type"].get(qt, {}).get("accuracy", float("nan")) if ref_acc else float("nan")
            t   = tcot_acc["per_type"].get(qt, {}).get("accuracy", float("nan"))
            d   = t - ref if (ref == ref and t == t) else float("nan")
            sign = "+" if d >= 0 else ""
            ref_s = f"{ref:.1f}%" if ref == ref else "  n/a"
            t_s   = f"{t:.1f}%"   if t   == t   else "  n/a"
            d_s   = f"{sign}{d:.1f}pp" if d == d else "   n/a"
            print(f"  {qt:<32} {ref_s:>8} {t_s:>8} {d_s:>8}")

    print("=" * 75)


def main():
    from utils.results_io import build_run_tag
    tcot_results = load_all_results()
    print_report(tcot_results)

    # Load baseline results — tags must match what each script wrote
    baseline_results = load_all_results(
        variant="baseline",
        tag=build_run_tag("baseline"),
    )
    native_results = load_all_results(
        variant="baseline_native",
        tag=build_run_tag("baseline_native"),
    )

    if baseline_results or native_results:
        print_comparison(tcot_results, baseline_results, native_results)
    else:
        print("\n  (No baseline results found. "
              "Run baseline.py and/or baseline_native.py first.)")

    # Optionally save a summary JSON
    acc_data = compute_accuracy(tcot_results)
    sel_data = compute_selection_stats(tcot_results)
    summary = {
        "dataset" : config.DATASET,
        "model"   : config.MODEL,
        "variant" : config.TCOT_VARIANT,
        "n"       : acc_data["total"],
        "accuracy": acc_data["accuracy"],
        **sel_data,
        "per_type": acc_data["per_type"],
    }

    if baseline_results:
        base_acc = compute_accuracy(baseline_results)
        summary["baseline_uniform_accuracy"] = base_acc["accuracy"]
        summary["delta_vs_uniform_baseline"] = acc_data["accuracy"] - base_acc["accuracy"]
    if native_results:
        nat_acc = compute_accuracy(native_results)
        summary["baseline_native_accuracy"] = nat_acc["accuracy"]
        summary["delta_vs_native_baseline"] = acc_data["accuracy"] - nat_acc["accuracy"]

    summary_dir = os.path.join(config.RESULTS_DIR, config.DATASET)
    os.makedirs(summary_dir, exist_ok=True)
    model_slug  = config.MODEL.replace("/", "-").replace(" ", "_")
    summary_path = os.path.join(
        summary_dir,
        f"{model_slug}_{config.TCOT_VARIANT}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()