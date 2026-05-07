#!/usr/bin/env python3
"""
length_control_chosen.py — produce a length-controlled variant of a DPO dataset.

For each pair, truncate `chosen` to <= MAX_WORDS words at the nearest sentence
boundary (or hard-cut if no boundary found). `rejected` and everything else are
left untouched. Drops pairs where chosen<2 words or chosen==rejected after trim.

The point: removes the verbosity-as-reward shortcut. With chosen and rejected
length-matched, the policy can't game the reward by writing longer answers; it
has to actually be more right.
"""
import json, sys, re, argparse
from pathlib import Path

def truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.rstrip()
    cut = " ".join(words[:max_words]).rstrip()
    # Prefer to cut at the last sentence/clause break in the truncated chunk.
    m = list(re.finditer(r"[.!?]\s|[.!?]$", cut))
    if m and m[-1].end() >= int(max_words * 0.6 * 6):  # ~ at least 60% of cut
        cut = cut[: m[-1].end()].rstrip()
    return cut

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--max-words", type=int, default=60,
                    help="upper bound on chosen length (median SFT output is ~36)")
    args = ap.parse_args()

    data = json.load(open(args.in_file))
    out = []
    n_trimmed = 0
    n_dropped = 0
    pre_lens = []
    post_lens = []
    for r in data:
        ch = r.get("chosen", "")
        rj = r.get("rejected", "")
        pre_lens.append(len(ch.split()))
        new_ch = truncate(ch, args.max_words)
        if len(new_ch.split()) < 2:
            n_dropped += 1
            continue
        if new_ch.strip() == rj.strip():
            n_dropped += 1
            continue
        if new_ch != ch:
            n_trimmed += 1
        post_lens.append(len(new_ch.split()))
        nr = dict(r)
        nr["chosen"] = new_ch
        out.append(nr)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_file, "w"), indent=2)

    def stats(xs):
        s = sorted(xs)
        n = len(s)
        return f"n={n} mean={sum(s)/n:.1f} med={s[n//2]} p90={s[int(0.9*n)]} max={s[-1]}"
    print(f"in : {len(data)} pairs   {stats(pre_lens)}")
    print(f"out: {len(out)} pairs   {stats(post_lens)}")
    print(f"  trimmed: {n_trimmed}   dropped: {n_dropped}   max_words={args.max_words}")
    print(f"  -> {args.out_file}")

if __name__ == "__main__":
    main()
