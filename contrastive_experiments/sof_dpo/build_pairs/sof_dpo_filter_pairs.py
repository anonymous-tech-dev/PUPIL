"""
sof_dpo_filter_pairs.py — Drop pairs whose `rejected` is already substantially
correct (the ablation accidentally produced a near-GT answer).

Two-stage cheap+thorough filter:
  1. ROUGE-L(rejected, chosen) > rouge_thresh  -> reject (too similar to GT)
  2. Numeric/keyword overlap rule              -> reject if rejected contains
                                                  ALL salient tokens of GT
After this script you pass survivors to the LLM judge (separate step) or skip
the judge for cost reasons.  The judge call is intentionally NOT included here
because it depends on which API you're using; we emit the prompts ready to go.

CLI
---
    python sof_dpo_filter_pairs.py \\
        --in-glob "../data/negatives_qwen3vl8b/negatives_*.shard*.jsonl" \\
        --out-jsonl ../data/pairs_after_filter.jsonl \\
        --judge-prompts-jsonl ../data/judge_prompts.jsonl \\
        --rouge-thresh 0.55
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

# --- Lightweight ROUGE-L ---------------------------------------------------
_TOK = re.compile(r"[A-Za-z0-9]+")


def _toks(s: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(s or "")]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    if n * m > 2_000_000:  # cap pathological cases
        a, b = a[:1000], b[:1000]
        n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def rouge_l(hyp: str, ref: str) -> float:
    h, r = _toks(hyp), _toks(ref)
    if not h or not r:
        return 0.0
    lcs = _lcs_len(h, r)
    if lcs == 0:
        return 0.0
    p = lcs / len(h)
    rec = lcs / len(r)
    if p + rec == 0:
        return 0.0
    return 2 * p * rec / (p + rec)


# --- Salient-token overlap (numbers, units, capitalised proper nouns) ------
_NUM = re.compile(r"\b\d+(?:\.\d+)?(?:e-?\d+)?\b")
_PROPER = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")


def salient_tokens(s: str) -> set[str]:
    s = s or ""
    return set(_NUM.findall(s)) | set(t.lower() for t in _PROPER.findall(s))


def keyword_already_in_rejected(chosen: str, rejected: str,
                                cover_thresh: float = 0.85) -> bool:
    keys = salient_tokens(chosen)
    if len(keys) < 3:
        return False
    rej_keys = salient_tokens(rejected)
    cover = len(keys & rej_keys) / max(1, len(keys))
    return cover >= cover_thresh


JUDGE_TEMPLATE = (
    "You are a strict grader. Decide whether the CANDIDATE answer is "
    "semantically equivalent to the REFERENCE answer for the given QUESTION.\n\n"
    "QUESTION:\n{q}\n\n"
    "REFERENCE:\n{ref}\n\n"
    "CANDIDATE:\n{cand}\n\n"
    "Reply with exactly one word: YES, PARTIAL, or NO."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--judge-prompts-jsonl", default=None,
                    help="If set, emit one judge prompt per surviving pair.")
    ap.add_argument("--rouge-thresh", type=float, default=0.55)
    ap.add_argument("--keyword-cover-thresh", type=float, default=0.85)
    args = ap.parse_args()

    files = sorted(glob.glob(args.in_glob))
    if not files:
        raise SystemExit(f"No files match: {args.in_glob}")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    judge_path = Path(args.judge_prompts_jsonl) if args.judge_prompts_jsonl else None
    if judge_path:
        judge_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    counts_per_axis: dict[str, Counter] = {}
    surviving = 0
    seen_ids: set[str] = set()
    fout = open(out_path, "w")
    fjud = open(judge_path, "w") if judge_path else None
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ax = rec["axis"]
                counts_per_axis.setdefault(ax, Counter())
                counts_per_axis[ax]["total"] += 1
                counts["total"] += 1
                qid_ax = f"{rec['query_id']}::{ax}"
                if qid_ax in seen_ids:
                    counts["dup"] += 1
                    continue
                seen_ids.add(qid_ax)

                rouge = rouge_l(rec["rejected"], rec["chosen"])
                if rouge >= args.rouge_thresh:
                    counts_per_axis[ax]["drop_rouge"] += 1
                    counts["drop_rouge"] += 1
                    continue
                if keyword_already_in_rejected(rec["chosen"], rec["rejected"],
                                                args.keyword_cover_thresh):
                    counts_per_axis[ax]["drop_keyword"] += 1
                    counts["drop_keyword"] += 1
                    continue

                rec["filter_rouge"] = round(rouge, 4)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if fjud:
                    jp = {
                        "query_id": rec["query_id"],
                        "axis": ax,
                        "prompt": JUDGE_TEMPLATE.format(
                            q=rec["question"], ref=rec["chosen"],
                            cand=rec["rejected"]),
                    }
                    fjud.write(json.dumps(jp, ensure_ascii=False) + "\n")
                surviving += 1
                counts_per_axis[ax]["kept"] += 1
                counts["kept"] += 1

    fout.close()
    if fjud:
        fjud.close()

    print(f"\n=== Filter summary ===")
    print(f"  total in : {counts['total']}")
    print(f"  duplicates dropped: {counts['dup']}")
    print(f"  dropped (rouge>={args.rouge_thresh}):   {counts['drop_rouge']}")
    print(f"  dropped (keyword cover):                {counts['drop_keyword']}")
    print(f"  surviving (-> {out_path.name}):         {counts['kept']}")
    print("\n  per-axis:")
    for ax, c in sorted(counts_per_axis.items()):
        kept_pct = 100 * c.get("kept", 0) / max(1, c.get("total", 1))
        print(f"    {ax:8s}  total={c['total']:4d}  kept={c.get('kept',0):4d}  "
              f"drop_rouge={c.get('drop_rouge',0):4d}  drop_kw={c.get('drop_keyword',0):4d}  "
              f"kept%={kept_pct:5.1f}")
    if fjud:
        print(f"\n  judge prompts written: {judge_path}")


if __name__ == "__main__":
    main()
