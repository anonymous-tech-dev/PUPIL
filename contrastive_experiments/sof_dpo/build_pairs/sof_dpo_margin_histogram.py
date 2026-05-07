"""
sof_dpo_margin_histogram.py — Quick diagnostic plot/text histogram of
ref_margin_per_tok across pairs, broken down by SoF axis.  No matplotlib;
prints an ASCII histogram so it works on any node.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    by_axis: dict[str, list[float]] = defaultdict(list)
    all_vals: list[float] = []
    for fp in sorted(glob.glob(args.in_glob)):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                v = r.get("ref_margin_per_tok")
                if v is None:
                    continue
                by_axis[r["axis"]].append(v)
                all_vals.append(v)
    if not all_vals:
        raise SystemExit("No values found.")

    lo, hi = min(all_vals), max(all_vals)
    rng = max(1e-6, hi - lo)
    width = rng / args.bins

    def hist(vals: list[float]) -> str:
        bins = [0] * args.bins
        for v in vals:
            i = min(args.bins - 1, int((v - lo) / width))
            bins[i] += 1
        peak = max(bins) or 1
        lines = []
        for i, c in enumerate(bins):
            edge = lo + i * width
            bar = "#" * int(40 * c / peak)
            lines.append(f"  {edge:+7.3f} | {bar} ({c})")
        return "\n".join(lines)

    print(f"\nALL pairs (n={len(all_vals)})  range=[{lo:+.3f}, {hi:+.3f}]")
    print(hist(all_vals))
    pos = sum(1 for v in all_vals if v > 0)
    print(f"  fraction with margin > 0   : {pos/len(all_vals):.3f}")
    print(f"  fraction with margin > +0.5: "
          f"{sum(1 for v in all_vals if v > 0.5)/len(all_vals):.3f}")
    print(f"  fraction with margin > +1.0: "
          f"{sum(1 for v in all_vals if v > 1.0)/len(all_vals):.3f}")

    for ax, vs in sorted(by_axis.items()):
        if not vs:
            continue
        print(f"\n{ax}  (n={len(vs)})  mean={sum(vs)/len(vs):+.3f}  "
              f"range=[{min(vs):+.3f}, {max(vs):+.3f}]")
        print(hist(vs))


if __name__ == "__main__":
    main()
