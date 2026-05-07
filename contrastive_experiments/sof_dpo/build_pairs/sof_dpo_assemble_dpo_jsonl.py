"""
sof_dpo_assemble_dpo_jsonl.py — Take per-shard pairs-with-margin JSONLs and
produce the final, deduped, optionally-saturation-filtered, optionally-balanced
DPO training file in the schema expected by the (legacy) DPODataset:

    {
      "id":     <query_id>,
      "video":  <abs path>,                  # the FULL video
      "prompt": "<video>\\nTranscript:\\n...\\n\\nQuestion: ...",
      "chosen": <ground_truth>,
      "rejected": <ablated answer>,
      "axis":   <visual|audio|time|priority>,
      "cognitive_category": <...>,
      "ref_margin": ..., "ref_margin_per_tok": ...
    }

Notes on the prompt
-------------------
The DPO trainer scores chosen and rejected under the SAME context, so we
construct a single FULL-CONTEXT prompt: video + transcript + question.  This
is the same context the margin-scorer used, ensuring DPO loss is consistent
with the saturation histogram we report.

CLI
---
    python sof_dpo_assemble_dpo_jsonl.py \\
        --in-glob "../data/pairs_with_margin*.jsonl" \\
        --out-train ../data/sof_dpo_train.json \\
        --out-val   ../data/sof_dpo_train.val.json \\
        --val-frac 0.05 --seed 0 \\
        --max-margin-per-tok 0.5     # drop saturated pairs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_pairs._io_utils import transcript_text  # noqa: E402


def build_full_prompt(question: str, transcript: str) -> str:
    if transcript:
        return (
            "<video>\n"
            f"Transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            f"Question: {question}"
        )
    return f"<video>\nQuestion: {question}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", default=None)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-margin-per-tok", type=float, default=None,
                    help="Drop pairs whose per-token margin already exceeds this "
                         "(saturated under reference policy).")
    ap.add_argument("--transcript-max-chars", type=int, default=6000)
    ap.add_argument("--cap-per-axis", type=int, default=None,
                    help="If set, downsample each axis to at most this count.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.in_glob))
    if not files:
        raise SystemExit(f"No files match: {args.in_glob}")

    pairs: list[dict] = []
    seen: set[str] = set()
    drop_sat = 0
    drop_dup = 0
    drop_no_video = 0
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = f"{rec['query_id']}::{rec['axis']}"
                if key in seen:
                    drop_dup += 1
                    continue
                seen.add(key)
                if (args.max_margin_per_tok is not None
                        and rec.get("ref_margin_per_tok", -1e9)
                        > args.max_margin_per_tok):
                    drop_sat += 1
                    continue
                if not rec.get("video_path") or not os.path.exists(rec["video_path"]):
                    drop_no_video += 1
                    continue
                pairs.append(rec)

    # Per-axis cap (uniformise mix; do BEFORE building prompts)
    if args.cap_per_axis is not None:
        rnd = random.Random(args.seed)
        by_ax: dict[str, list[dict]] = defaultdict(list)
        for r in pairs:
            by_ax[r["axis"]].append(r)
        capped = []
        for ax, lst in by_ax.items():
            rnd.shuffle(lst)
            capped.extend(lst[: args.cap_per_axis])
        pairs = capped

    rnd = random.Random(args.seed)
    rnd.shuffle(pairs)

    # Build prompt with full transcript
    out_records = []
    for rec in pairs:
        stub = {"video_path": rec["video_path"]}
        tx = transcript_text(stub, max_chars=args.transcript_max_chars)
        prompt = build_full_prompt(rec["question"], tx)
        out_records.append({
            "id": rec["query_id"],
            "video": rec["video_path"],
            "prompt": prompt,
            "chosen": rec["chosen"],
            "rejected": rec["rejected"],
            "axis": rec["axis"],
            "cognitive_category": rec.get("cognitive_category", ""),
            "ref_margin": rec.get("ref_margin"),
            "ref_margin_per_tok": rec.get("ref_margin_per_tok"),
            "filter_rouge": rec.get("filter_rouge"),
        })

    if args.out_val and args.val_frac > 0:
        n_val = max(1, int(len(out_records) * args.val_frac))
        val = out_records[:n_val]
        train = out_records[n_val:]
    else:
        val, train = [], out_records

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w") as f:
        json.dump(train, f, indent=2)
    print(f"Wrote {len(train)} train pairs -> {args.out_train}")
    if val:
        with open(args.out_val, "w") as f:
            json.dump(val, f, indent=2)
        print(f"Wrote {len(val)} val pairs   -> {args.out_val}")

    print(f"\n=== Assemble summary ===")
    print(f"  duplicates dropped     : {drop_dup}")
    print(f"  saturated   dropped    : {drop_sat}  "
          f"(thresh={args.max_margin_per_tok})")
    print(f"  no-video    dropped    : {drop_no_video}")
    by_ax_n: Counter = Counter(r["axis"] for r in out_records)
    for ax, n in sorted(by_ax_n.items()):
        print(f"  axis {ax:8s}  : {n}")


if __name__ == "__main__":
    main()
