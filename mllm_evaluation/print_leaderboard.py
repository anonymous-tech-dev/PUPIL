#!/usr/bin/env python3
"""
print_leaderboard.py — quick stdout leaderboard for Pupil.

Scans ``mllm_evaluation/results/<model>/<experiment>/`` for ``*_results.json``
files, computes judge accuracy from the ``judge_verdict`` field, and prints a
leaderboard. No markdown, no files written.

Usage:
    python print_leaderboard.py                        # all models, all experiments
    python print_leaderboard.py --experiment final_1k_benchmark
    python print_leaderboard.py --model qwen3_vl_ft    # only show one model's runs
    python print_leaderboard.py --qwen-only            # only Qwen3-VL-8B + variants (qwen3_vl, qwen3_vl_ft, ...)
    python print_leaderboard.py --breakdown            # also show per-pipeline / per-cognitive
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "ablation_results")


def _load_records(exp_dir: str):
    records = []
    for jf in glob.glob(os.path.join(exp_dir, "**", "*_results.json"), recursive=True):
        if "parity" in jf or "_shard" in jf:
            continue
        try:
            with open(jf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)

    # The true primary key is (query_id, question) — collapse defensively in
    # case shard merges left dups.  Keeps the LAST occurrence (latest judge).
    seen = {}
    for r in records:
        qid = r.get("query_id") or ""
        q = (r.get("question") or "").strip()
        seen[(qid, q) if qid or q else id(r)] = r
    return list(seen.values())


def _accuracy(records):
    correct = sum(1 for r in records if r.get("judge_verdict") is True)
    judged  = sum(1 for r in records if r.get("judge_verdict") is not None)
    pending = len(records) - judged
    return correct, judged, pending


def _grouped(records, key_fn):
    buckets = defaultdict(lambda: [0, 0])  # [correct, judged]
    for r in records:
        v = r.get("judge_verdict")
        if v is None:
            continue
        k = key_fn(r) or "unknown"
        buckets[k][1] += 1
        if v:
            buckets[k][0] += 1
    return dict(buckets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--model", default=None, help="Only include this model.")
    ap.add_argument("--experiment", default=None,
                    help="Only include this experiment subfolder name.")
    ap.add_argument("--qwen-only", action="store_true",
                    help="Only show Qwen3-VL-8B and its variants "
                         "(qwen3_vl, qwen3_vl_ft, qwen3_vl_matched, …).")
    ap.add_argument("--breakdown", action="store_true",
                    help="Also print per-pipeline-mode and per-category tables.")
    ap.add_argument("--include-noisy", action="store_true",
                    help="Include qwen3_vl_ft runs whose experiment name does not contain NOTX (hidden by default to keep the board readable).")
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"❌ results_dir not found: {args.results_dir}")
        return

    # Matches: qwen3_vl, qwen3_vl_ft, qwen3_vl_matched, qwen3_vl_<anything>.
    # Does NOT match: qwen32_vl, qwen2_5_vl, etc.
    QWEN_RE = re.compile(r"^qwen3_vl(?:_.*)?$")

    rows = []  # (model, experiment, correct, judged, pending)
    breakdown_data = {}  # (model, experiment) -> dict
    for model in sorted(os.listdir(args.results_dir)):
        if args.model and model != args.model:
            continue
        if args.qwen_only and not QWEN_RE.match(model):
            continue
        model_dir = os.path.join(args.results_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for exp in sorted(os.listdir(model_dir)):
            if args.experiment and exp != args.experiment:
                continue
            # Hide noisy non-NOTX qwen3_vl_ft runs from the default leaderboard.
            # Override with --include-noisy or by passing --experiment explicitly.
            # if (
            #     model == "qwen3_vl_ft"
            #     and "NOTX" not in exp
            #     and not args.include_noisy
            #     and not args.experiment
            # ):
            #     continue
            exp_dir = os.path.join(model_dir, exp)
            if not os.path.isdir(exp_dir):
                continue
            recs = _load_records(exp_dir)
            if not recs:
                continue
            c, j, p = _accuracy(recs)
            rows.append((model, exp, c, j, p))
            if args.breakdown:
                breakdown_data[(model, exp)] = {
                    "pipeline": _grouped(recs, lambda r: r.get("source_of_fact")),
                    "category": _grouped(recs, lambda r: r.get("category")),
                }

    if not rows:
        print("❌ No results found. Did the judge run?")
        print(f"   Looked under: {args.results_dir}")
        if args.model: print(f"   Filter model: {args.model}")
        if args.experiment: print(f"   Filter experiment: {args.experiment}")
        if args.qwen_only: print(f"   Filter qwen-only: matches ^qwen3_vl(_.*)?$")
        return

    # sort: judged>0 by accuracy desc, then unjudged
    def _key(r):
        m, e, c, j, p = r
        return (-(c / j) if j else 1.0,)
    rows.sort(key=_key)

    # widths
    mw = max(len(r[0]) for r in rows)
    ew = max(len(r[1]) for r in rows)
    mw = max(mw, len("Model"))
    ew = max(ew, len("Experiment"))

    print("\n🏆 Pupil leaderboard" + ("  (Qwen3-VL-8B + variants only)" if args.qwen_only else ""))
    print("=" * (mw + ew + 40))
    print(f"  #  {'Model':<{mw}}  {'Experiment':<{ew}}  {'Acc':>7}  {'Correct':>9}  {'Pending':>8}")
    print("  " + "-" * (mw + ew + 38))
    for i, (m, e, c, j, p) in enumerate(rows, 1):
        acc = f"{100*c/j:.1f}%" if j else "—"
        cor = f"{c}/{j}" if j else "0/0"
        print(f"  {i:>2} {m:<{mw}}  {e:<{ew}}  {acc:>7}  {cor:>9}  {p:>8}")
    print()

    if args.breakdown:
        for (m, e), d in breakdown_data.items():
            print(f"\n── {m}  /  {e} ──")
            for label, dd in (("Pipeline mode", d["pipeline"]),
                              ("Cognitive cat", d["category"])):
                if not dd:
                    continue
                print(f"  {label:<14} {'Acc':>7}  {'n':>6}")
                for k in sorted(dd, key=lambda x: -dd[x][0] / max(dd[x][1], 1)):
                    cc, tt = dd[k]
                    print(f"    {k:<25} {100*cc/tt:>6.1f}%  {tt:>6}")


if __name__ == "__main__":
    main()
