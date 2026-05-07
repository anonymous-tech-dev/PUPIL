"""
Paired McNemar test on per-qid GPT-judge correctness for two runs.

Usage:
    python tools/mcnemar_compare.py <run_A_gpt5_judge.json> <run_B_gpt5_judge.json>

Prints: contingency table (b/c), McNemar chi-square (with continuity correction),
exact binomial p-value, signed-rank-style summary, and the per-category breakdown.
"""

import json
import sys
import math
from collections import defaultdict


def load(path):
    with open(path) as f:
        d = json.load(f)
    samples = d.get("samples", [])
    by_id = {}
    for s in samples:
        qid = s.get("id")
        if qid is None:
            continue
        by_id[str(qid)] = s
    return d, by_id


def exact_binomial_two_sided(b, c):
    """Exact two-sided binomial test for McNemar (n = b + c, k = min(b, c), p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5), then doubled for two-sided
    log_total = -n * math.log(2)
    cum = 0.0
    for i in range(k + 1):
        # binomial coefficient via lgamma
        log_choose = (
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        )
        cum += math.exp(log_choose + log_total)
    p = min(1.0, 2.0 * cum)
    return p


def main():
    if len(sys.argv) != 3:
        print("Usage: python mcnemar_compare.py <A.json> <B.json>")
        sys.exit(2)

    path_a, path_b = sys.argv[1], sys.argv[2]
    da, ia = load(path_a)
    db, ib = load(path_b)

    common = sorted(set(ia.keys()) & set(ib.keys()))
    only_a = set(ia.keys()) - set(ib.keys())
    only_b = set(ib.keys()) - set(ia.keys())

    print("=" * 78)
    print("McNemar paired test")
    print("=" * 78)
    print(f"A: {path_a}")
    print(f"   total={len(ia)}  acc={da.get('accuracy', float('nan'))*100:.2f}%")
    print(f"B: {path_b}")
    print(f"   total={len(ib)}  acc={db.get('accuracy', float('nan'))*100:.2f}%")
    print(f"Common qids: {len(common)}  (A-only: {len(only_a)}, B-only: {len(only_b)})")
    print()

    a_only_correct = 0  # A correct, B wrong  (b)
    b_only_correct = 0  # B correct, A wrong  (c)
    both_correct = 0
    both_wrong = 0

    per_cat = defaultdict(lambda: {"b": 0, "c": 0, "both_right": 0, "both_wrong": 0, "n": 0})

    for qid in common:
        sa = ia[qid]
        sb = ib[qid]
        ca = bool(sa.get("is_correct"))
        cb = bool(sb.get("is_correct"))
        cat = sa.get("sub_category", "Unknown")
        per_cat[cat]["n"] += 1
        if ca and cb:
            both_correct += 1
            per_cat[cat]["both_right"] += 1
        elif (not ca) and (not cb):
            both_wrong += 1
            per_cat[cat]["both_wrong"] += 1
        elif ca and not cb:
            a_only_correct += 1  # b
            per_cat[cat]["b"] += 1
        else:
            b_only_correct += 1  # c
            per_cat[cat]["c"] += 1

    n = len(common)
    acc_a_paired = (both_correct + a_only_correct) / n if n else float("nan")
    acc_b_paired = (both_correct + b_only_correct) / n if n else float("nan")

    print("Contingency on COMMON qids:")
    print(f"                        B correct   B wrong")
    print(f"  A correct          {both_correct:>6d}     {a_only_correct:>6d}    (A correct total = {both_correct + a_only_correct})")
    print(f"  A wrong            {b_only_correct:>6d}     {both_wrong:>6d}    (A wrong   total = {b_only_correct + both_wrong})")
    print()
    print(f"Paired accuracy:  A = {acc_a_paired*100:.2f}%   B = {acc_b_paired*100:.2f}%   Δ(B-A) = {(acc_b_paired - acc_a_paired)*100:+.2f}pp")
    print(f"Discordant pairs: A-only-correct b={a_only_correct}, B-only-correct c={b_only_correct}, n_disc={a_only_correct + b_only_correct}")
    print()

    b = a_only_correct
    c = b_only_correct
    n_disc = b + c
    if n_disc == 0:
        print("No discordant pairs — runs are perfectly identical on this set.")
        return
    # McNemar chi-square with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # Exact two-sided binomial p-value (preferred when n_disc small)
    p_exact = exact_binomial_two_sided(b, c)
    # Approx normal z (without continuity)
    z = (c - b) / math.sqrt(b + c)

    print(f"McNemar χ² (continuity corrected) = {chi2:.4f}")
    print(f"Approx z (B vs A, no correction)  = {z:+.3f}")
    print(f"Exact binomial p-value (two-sided) = {p_exact:.4g}")
    sig = ""
    if p_exact < 0.001:
        sig = "*** (p < 0.001)"
    elif p_exact < 0.01:
        sig = "** (p < 0.01)"
    elif p_exact < 0.05:
        sig = "* (p < 0.05)"
    else:
        sig = "n.s. (p ≥ 0.05)"
    print(f"Significance: {sig}")

    # 95% CI for the proportion of discordant pairs that B wins
    if n_disc > 0:
        p_hat = c / n_disc
        # Wilson interval
        z_ = 1.96
        denom = 1 + z_ ** 2 / n_disc
        center = (p_hat + z_ ** 2 / (2 * n_disc)) / denom
        half = (z_ * math.sqrt(p_hat * (1 - p_hat) / n_disc + z_ ** 2 / (4 * n_disc ** 2))) / denom
        lo, hi = center - half, center + half
        # convert to delta in pp on full sample
        delta_lo = (2 * lo - 1) * n_disc / n
        delta_hi = (2 * hi - 1) * n_disc / n
        print(f"95% CI for Δ(B-A) on full sample: [{delta_lo*100:+.2f}pp, {delta_hi*100:+.2f}pp]")

    # Per-category breakdown
    print()
    print("Per sub-category (sorted by |b-c|):")
    rows = []
    for cat, d in per_cat.items():
        rows.append((abs(d["b"] - d["c"]), cat, d))
    rows.sort(reverse=True)
    print(f"  {'sub_category':<28} {'n':>4}  {'A>B':>5} {'B>A':>5} {'Δ(B-A)':>8}")
    for _, cat, d in rows:
        delta = (d["c"] - d["b"]) / d["n"] * 100 if d["n"] else 0
        print(f"  {cat:<28} {d['n']:>4}  {d['b']:>5} {d['c']:>5}  {delta:+7.2f}pp")


if __name__ == "__main__":
    main()
