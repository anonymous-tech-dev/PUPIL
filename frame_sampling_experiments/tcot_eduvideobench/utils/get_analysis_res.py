"""
analyze_results.py — Diagnostic analysis of TCoT JSONL results.

Computes statistics that reveal WHY TCoT underperforms the native baseline:
  1. Context size distribution (num_context vs k budget)
  2. Parse failure rate per result file
  3. Selection aggressiveness (frames selected per segment)
  4. Accuracy by question type
  5. Accuracy by num_context bucket
  6. Raw answer format distribution
  7. Per-video accuracy (to spot systematic failures)

Usage:
    python analyze_results.py --dir results/lvbench_v2
    python analyze_results.py --file results/lvbench_v2/Qwen2.5-VL-7B_dynamic_segment_l12_s64_k1024_u56_results.jsonl
"""

import json
import os
import sys
import argparse
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def accuracy(results):
    correct = sum(1 for r in results
                  if r.get("predicted_letter") and r.get("ground_truth")
                  and r["predicted_letter"] == r["ground_truth"])
    return correct, len(results), 100.0 * correct / len(results) if results else 0.0


def print_separator(title=""):
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("─" * w)


def analyze_file(path: str):
    
    results = load_jsonl(path)
    if not results:
        print(f"  [empty] {path}")
        return

    fname = os.path.basename(path)
    print(f"\n{'='*70}")
    print(f"FILE: {fname}")
    print(f"{'='*70}")

    c, t, acc = accuracy(results)
    print(f"Overall accuracy: {acc:.2f}%  ({c}/{t})")

    # ── Detect config from results ────────────────────────────────────────
    stage = results[0].get("stage", "unknown")
    print(f"Stage: {stage}")

    # ── 1. Context size distribution ──────────────────────────────────────
    print_separator("1. CONTEXT SIZE (num_context passed to answerer)")
    num_contexts = [r["num_context"] for r in results if "num_context" in r]
    if num_contexts:
        import statistics
        print(f"  Mean  : {statistics.mean(num_contexts):.1f}")
        print(f"  Median: {statistics.median(num_contexts):.1f}")
        print(f"  Min   : {min(num_contexts)}")
        print(f"  Max   : {max(num_contexts)}")
        print(f"  Stdev : {statistics.stdev(num_contexts):.1f}" if len(num_contexts) > 1 else "")

        # Bucket distribution
        buckets = Counter()
        for n in num_contexts:
            if n < 50:      buckets["<50"] += 1
            elif n < 100:   buckets["50-99"] += 1
            elif n < 200:   buckets["100-199"] += 1
            elif n < 300:   buckets["200-299"] += 1
            elif n < 500:   buckets["300-499"] += 1
            elif n < 700:   buckets["500-699"] += 1
            elif n < 900:   buckets["700-899"] += 1
            elif n < 1024:  buckets["900-1023"] += 1
            else:           buckets[">=1024"] += 1

        for label in ["<50","50-99","100-199","200-299","300-499","500-699","700-899","900-1023",">=1024"]:
            if buckets[label]:
                bar = "█" * (buckets[label] * 30 // len(num_contexts))
                print(f"  {label:>10}: {buckets[label]:4d} ({100*buckets[label]/len(num_contexts):5.1f}%) {bar}")

    # ── 2. Parse failure rate ─────────────────────────────────────────────
    print_separator("2. PARSE FAILURE RATE (selection call)")
    if stage != "baseline" and stage != "baseline_native":
        total_parse_checks = 0
        total_parse_fails = 0
        total_empty_fails = 0
        samples_with_any_fail = 0

        for r in results:
            justs = r.get("justifications", [])
            had_fail = False
            for j in justs:
                total_parse_checks += 1
                if "PARSE FAILED" in j:
                    total_parse_fails += 1
                    had_fail = True
                elif "EMPTY SELECTION" in j:
                    total_empty_fails += 1
                    had_fail = True
            if had_fail:
                samples_with_any_fail += 1

        print(f"  Samples with ≥1 parse/empty failure: {samples_with_any_fail}/{t} ({100*samples_with_any_fail/t:.1f}%)")
        if total_parse_checks > 0:
            print(f"  Total segment calls checked: {total_parse_checks}")
            print(f"  PARSE FAILED fallbacks: {total_parse_fails} ({100*total_parse_fails/total_parse_checks:.1f}%)")
            print(f"  EMPTY SELECTION fallbacks: {total_empty_fails} ({100*total_empty_fails/total_parse_checks:.1f}%)")
    else:
        print("  N/A (baseline — no selection calls)")

    # ── 3. Selection aggressiveness ───────────────────────────────────────
    print_separator("3. SELECTION AGGRESSIVENESS")
    if stage not in ("baseline", "baseline_native"):
        num_selected_list = [r["num_selected"] for r in results if "num_selected" in r]
        if num_selected_list:
            import statistics
            print(f"  Frames selected (num_selected):")
            print(f"    Mean  : {statistics.mean(num_selected_list):.1f}")
            print(f"    Median: {statistics.median(num_selected_list):.1f}")
            print(f"    Min   : {min(num_selected_list)}")
            print(f"    Max   : {max(num_selected_list)}")

        pct_selected_list = [r["pct_selected"] for r in results
                             if r.get("pct_selected", -1) >= 0]
        if pct_selected_list:
            import statistics
            print(f"  % of total frames selected:")
            print(f"    Mean  : {statistics.mean(pct_selected_list):.1f}%")
            print(f"    Median: {statistics.median(pct_selected_list):.1f}%")

        # How often does num_selected << k?
        # Infer k from filename
        k_match = re.search(r"_k(\d+)", fname)
        k = int(k_match.group(1)) if k_match else None
        if k:
            u_match = re.search(r"_u(\d+)", fname)
            u = int(u_match.group(1)) if u_match else 56
            m = k - u
            under_budget = sum(1 for r in results if r.get("num_selected", 0) < m)
            print(f"  k={k}, u={u}, m=k-u={m}")
            print(f"  Samples where num_selected < m ({m}): {under_budget}/{t} ({100*under_budget/t:.1f}%)")
            print(f"  → These samples have answerer context BELOW budget k={k}")
    else:
        print("  N/A (baseline)")

    # ── 4. Accuracy by question type ──────────────────────────────────────
    print_separator("4. ACCURACY BY QUESTION TYPE")
    qt_stats = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in results:
        qtypes = r.get("question_type", [])
        if not qtypes:
            qtypes = ["unknown"]
        is_correct = (r.get("predicted_letter") and r.get("ground_truth")
                      and r["predicted_letter"] == r["ground_truth"])
        for qt in qtypes:
            qt_stats[qt][1] += 1
            if is_correct:
                qt_stats[qt][0] += 1

    for qt, (corr, tot) in sorted(qt_stats.items(), key=lambda x: -x[1][1]):
        a = 100.0 * corr / tot if tot else 0
        print(f"  {qt:<30}: {a:5.1f}%  ({corr}/{tot})")

    # ── 5. Accuracy by num_context bucket ─────────────────────────────────
    print_separator("5. ACCURACY BY NUM_CONTEXT BUCKET")
    bucket_stats = defaultdict(lambda: [0, 0])
    for r in results:
        n = r.get("num_context", 0)
        if n < 50:      b = "<50"
        elif n < 100:   b = "50-99"
        elif n < 200:   b = "100-199"
        elif n < 300:   b = "200-299"
        elif n < 500:   b = "300-499"
        elif n < 700:   b = "500-699"
        elif n < 900:   b = "700-899"
        elif n < 1024:  b = "900-1023"
        else:           b = ">=1024"
        bucket_stats[b][1] += 1
        if (r.get("predicted_letter") and r.get("ground_truth")
                and r["predicted_letter"] == r["ground_truth"]):
            bucket_stats[b][0] += 1

    for label in ["<50","50-99","100-199","200-299","300-499","500-699","700-899","900-1023",">=1024"]:
        if bucket_stats[label][1] > 0:
            corr, tot = bucket_stats[label]
            a = 100.0 * corr / tot
            print(f"  {label:>10}: {a:5.1f}%  ({corr}/{tot})")

    # ── 6. Raw answer format ──────────────────────────────────────────────
    print_separator("6. RAW ANSWER FORMAT (first 20 chars of raw_answer)")
    answer_fmt = Counter()
    empty_preds = 0
    for r in results:
        raw = r.get("raw_answer", "")
        if not r.get("predicted_letter"):
            empty_preds += 1
        # Categorise
        if not raw:
            answer_fmt["[empty]"] += 1
        elif re.match(r"^[A-E]$", raw.strip()):
            answer_fmt["single_letter"] += 1
        elif "Final Answer" in raw:
            answer_fmt["Final Answer: (X)"] += 1
        elif re.search(r"\([A-E]\)", raw):
            answer_fmt["contains (X)"] += 1
        else:
            answer_fmt["other"] += 1

    print(f"  Failed to extract letter: {empty_preds}/{t} ({100*empty_preds/t:.1f}%)")
    for fmt, cnt in answer_fmt.most_common():
        print(f"  {fmt:<25}: {cnt:4d} ({100*cnt/t:.1f}%)")

    # ── 7. Time stats ─────────────────────────────────────────────────────
    print_separator("7. TIMING")
    times = [r["time_taken_secs"] for r in results if "time_taken_secs" in r]
    if times:
        import statistics
        print(f"  Mean time/sample : {statistics.mean(times):.1f}s")
        print(f"  Median           : {statistics.median(times):.1f}s")
        print(f"  Total wall time  : {sum(times)/3600:.1f}h")


def compare_files(paths: List[str]):
    """Side-by-side accuracy + key stats comparison."""
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'File':<55} {'Acc':>6}  {'N':>5}  {'AvgCtx':>7}")
    print("─" * 70)
    for path in paths:
        results = load_jsonl(path)
        if not results:
            continue
        c, t, acc = accuracy(results)
        avg_ctx = (sum(r.get("num_context", 0) for r in results) / t) if t else 0
        fname = os.path.basename(path)
        # Shorten filename for display
        short = fname.replace("Qwen2.5-VL-7B_", "").replace("_results.jsonl", "")
        print(f"  {short:<53} {acc:6.2f}%  {t:5d}  {avg_ctx:7.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory containing JSONL result files")
    parser.add_argument("--file", type=str, default=None,
                        help="Single JSONL file to analyze")
    args = parser.parse_args()

    paths = []
    if args.file:
        # Enforce the _v1 condition even if a single file is passed directly
        if args.file.endswith(".jsonl"):
            paths = [args.file]
        else:
            print(f"Skipping '{args.file}' because it does not end with '_v1.jsonl'.")
    elif args.dir:
        paths = sorted([
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.endswith("_v2.jsonl")  # <-- Updated condition here
        ])
    else:
        # Default: look in results/
        for root, _, files in os.walk("results"):
            for f in files:
                if f.endswith(".jsonl"):  # <-- Updated condition here
                    paths.append(os.path.join(root, f))
        paths.sort()

    if not paths:
        print("No JSONL files ending in '_v1.jsonl' found.")
        sys.exit(1)

    print(f"Analyzing {len(paths)} file(s)")
    compare_files(paths)

    for path in paths:
        analyze_file(path)


if __name__ == "__main__":
    main()