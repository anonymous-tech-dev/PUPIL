#!/usr/bin/env python3
"""
apply_judge_to_dpo.py — Drop pairs whose judge verdict is YES or PARTIAL.

Reads:
    --judge   judge_results.jsonl  (from run_judge_parallel.py)
    --train   sof_dpo_train.json   (the assembled DPO file currently used)
    --val     sof_dpo_train.val.json
Writes:
    <train>.judged.json  and  <val>.judged.json
plus a side-by-side <train>.judged.stats.json with per-axis drop counts.

Drop policy:
    YES      -> drop  (the "rejected" answer is actually correct)
    PARTIAL  -> drop  (ambiguous — would inject DPO noise)
    NO       -> keep  (clean negative)
    ERROR    -> keep  (fail-open; the cheap filters already passed it)
    missing  -> keep  (same)
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter
from pathlib import Path


def load_judge(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["query_id"]] = r.get("verdict", "ERROR")
    return out


def filter_one(records: list[dict], verdicts: dict[str, str]) -> tuple[list[dict], Counter]:
    kept: list[dict] = []
    stats: Counter = Counter()
    for r in records:
        v = verdicts.get(r["id"], "MISSING")
        stats[f"verdict_{v}"] += 1
        stats[f"axis_{r.get('axis','?')}_seen"] += 1
        if v in ("YES", "PARTIAL"):
            stats[f"axis_{r.get('axis','?')}_dropped"] += 1
            continue
        kept.append(r)
    stats["in_total"] = len(records)
    stats["out_total"] = len(kept)
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-suffix", default=".judged.json")
    args = ap.parse_args()

    verdicts = load_judge(Path(args.judge))
    print(f"[judge] {len(verdicts)} verdicts loaded from {args.judge}")

    for tag, src in (("train", args.train), ("val", args.val)):
        src_p = Path(src)
        with src_p.open() as f:
            data = json.load(f)
        kept, stats = filter_one(data, verdicts)
        out_p = src_p.with_suffix("")  # drop .json
        out_p = Path(str(out_p) + args.out_suffix)
        with out_p.open("w") as f:
            json.dump(kept, f, ensure_ascii=False, indent=0)
        stat_p = out_p.with_suffix(out_p.suffix + ".stats.json")
        with stat_p.open("w") as f:
            json.dump(dict(stats), f, indent=2)
        print(
            f"[{tag}] {stats['in_total']} -> {stats['out_total']}  "
            f"(kept {stats['out_total']/max(1,stats['in_total']):.1%})  "
            f"-> {out_p}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
