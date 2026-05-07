"""
sof_dpo_filter_pairs_v2.py — v2 filter that runs the same ROUGE + keyword
guards as v1, plus an explicit ABSTENTION drop (any rejected the v2 generator
flagged with `v2_final_abstain` AFTER 3 retries, plus a defensive re-check).

Also emits judge prompts using the SAME GPT-5 judge template that
mllm_evaluation/evaluate_parallel.py uses on the benchmark itself, so the
data-curation judge is consistent with the eval-time judge.

CLI
---
    python sof_dpo_filter_pairs_v2.py \
        --in-glob "../old_dpo_revised_data_8b/negatives_v2/final_*.shard*.jsonl" \
        --out-jsonl ../old_dpo_revised_data_8b/pairs_after_filter.jsonl \
        --judge-prompts-jsonl ../old_dpo_revised_data_8b/judge_prompts.jsonl \
        --rouge-thresh 0.55
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_pairs._abstain_utils import is_abstain  # noqa: E402

# --- Lightweight ROUGE-L (copied from v1) ---------------------------------
_TOK = re.compile(r"[A-Za-z0-9]+")


def _toks(s: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(s or "")]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    if n * m > 2_000_000:
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
    return 2 * p * rec / (p + rec) if p + rec else 0.0


# --- Salient-token overlap ------------------------------------------------
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


# --- Judge prompt: identical wording to mllm_evaluation/evaluate_parallel.py
JUDGE_TEMPLATE = """
    You are an impartial judge evaluating a Multimodal AI's response.

    Question: "{q}"
    Ground Truth: "{ref}"
    Model Prediction: "{cand}"

    Task:
    1. Compare the factual core of the Model Prediction against the Ground Truth.
    2. Determine if the prediction matches the ground truth (ignore minor phrasing differences).
    3. Output strictly valid JSON.

    Format examples (illustrative only — do NOT copy their content):
    {{
        "reason": "Short explanation of why the prediction matches the ground truth...",
        "verdict": true
    }}
    {{
        "reason": "Short explanation of why the prediction does NOT match the ground truth...",
        "verdict": false
    }}
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--judge-prompts-jsonl", default=None)
    ap.add_argument("--rouge-thresh", type=float, default=0.55)
    ap.add_argument("--keyword-cover-thresh", type=float, default=0.85)
    ap.add_argument("--keep-abstentions", action="store_true",
                    help="DEBUG: skip the abstention drop (you almost never "
                         "want this).")
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
                    counts_per_axis[ax]["drop_dup"] += 1
                    continue
                seen_ids.add(qid_ax)

                rejected = rec.get("rejected", "")

                # 1. Drop abstentions (the whole point of v2).
                v2_flag = bool(rec.get("v2_final_abstain", False))
                regex_abst = is_abstain(rejected)
                if not args.keep_abstentions and (v2_flag or regex_abst):
                    counts["drop_abstain"] += 1
                    counts_per_axis[ax]["drop_abstain"] += 1
                    continue

                # 2. Drop trivially-empty rejections.
                if len(_toks(rejected)) < 3:
                    counts["drop_empty"] += 1
                    counts_per_axis[ax]["drop_empty"] += 1
                    continue

                # 3. ROUGE: too similar to the chosen → no signal.
                rouge = rouge_l(rejected, rec["chosen"])
                if rouge >= args.rouge_thresh:
                    counts["drop_rouge"] += 1
                    counts_per_axis[ax]["drop_rouge"] += 1
                    continue

                # 4. Keyword cover: rejected contains all salient tokens of GT.
                if keyword_already_in_rejected(rec["chosen"], rejected,
                                                args.keyword_cover_thresh):
                    counts["drop_keyword"] += 1
                    counts_per_axis[ax]["drop_keyword"] += 1
                    continue

                rec["filter_rouge"] = round(rouge, 4)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if fjud:
                    jp = {
                        "query_id": rec["query_id"],
                        "axis": ax,
                        "prompt": JUDGE_TEMPLATE.format(
                            q=rec["question"], ref=rec["chosen"],
                            cand=rejected),
                    }
                    fjud.write(json.dumps(jp, ensure_ascii=False) + "\n")
                counts["kept"] += 1
                counts_per_axis[ax]["kept"] += 1

    fout.close()
    if fjud:
        fjud.close()

    # ── Report ────────────────────────────────────────────────────────────
    print("\n=== v2 filter summary ===")
    print(f"  total in   : {counts['total']}")
    print(f"  duplicates : {counts['dup']}")
    print(f"  abstain    : {counts['drop_abstain']}    "
          f"(this is the v2 win — these would have been DPO poison)")
    print(f"  empty      : {counts['drop_empty']}")
    print(f"  rouge>={args.rouge_thresh:.2f}: {counts['drop_rouge']}")
    print(f"  keyword    : {counts['drop_keyword']}")
    print(f"  KEPT       : {counts['kept']}    -> {out_path}")
    if fjud:
        print(f"  judge prompts -> {judge_path}")
    print("\n  per-axis breakdown:")
    print(f"    {'axis':10s} {'total':>6s} {'kept':>6s} "
          f"{'absDrp':>7s} {'empDrp':>7s} {'rouge':>6s} {'kw':>5s} {'kept%':>7s}")
    for ax, c in sorted(counts_per_axis.items()):
        kept_pct = 100 * c.get("kept", 0) / max(1, c.get("total", 1))
        print(f"    {ax:10s} {c.get('total',0):6d} {c.get('kept',0):6d} "
              f"{c.get('drop_abstain',0):7d} {c.get('drop_empty',0):7d} "
              f"{c.get('drop_rouge',0):6d} {c.get('drop_keyword',0):5d} "
              f"{kept_pct:6.1f}%")


if __name__ == "__main__":
    main()
