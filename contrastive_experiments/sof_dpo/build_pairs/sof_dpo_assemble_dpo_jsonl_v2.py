"""
sof_dpo_assemble_dpo_jsonl_v2.py — v2 assembly:

  * No-transcript prompt by default (--with-transcript to opt back in).
    The leaderboard `*_NOTX_*` runs all consistently outperformed the
    with-transcript variants (transcripts are full of Whisper
    hallucinations on non-English fragments).
  * Prompt is "<video>\\nQuestion: {q}", which after replace_image_tokens()
    becomes "<|vision_start|><|video_pad|><|vision_end|>\\nQuestion: {q}".
    The eval-side wrappers (qwen3_vl_finetuned.py, qwen3_vl_matched.py)
    have been updated to land on the same post-template string.
  * Saturation cut, per-axis cap, dedup, val-split — same as v1.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def build_prompt(question: str, transcript: str = "") -> str:
    if transcript:
        return ("<video>\n"
                f"Transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
                f"Question: {question}")
    return f"<video>\nQuestion: {question}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", default=None)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-margin-per-tok", type=float, default=None,
                    help="Drop pairs whose per-token ref margin exceeds this "
                         "(saturated under reference policy).")
    ap.add_argument("--cap-per-axis", type=int, default=None,
                    help="If set, cap each axis to at most this count.")
    ap.add_argument("--with-transcript", action="store_true",
                    help="Include the full ASR transcript in the prompt "
                         "(NOT recommended — leaderboard says NOTX wins).")
    ap.add_argument("--transcript-max-chars", type=int, default=6000)
    args = ap.parse_args()

    if args.with_transcript:
        # Lazy import only when needed (saves a startup hit and keeps this
        # module importable in a context without _io_utils on PYTHONPATH).
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from build_pairs._io_utils import transcript_text  # noqa: E402

    files = sorted(glob.glob(args.in_glob))
    if not files:
        raise SystemExit(f"No files match: {args.in_glob}")

    pairs: list[dict] = []
    seen: set[str] = set()
    drop_sat = drop_dup = drop_no_video = 0
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

    out_records = []
    for rec in pairs:
        if args.with_transcript:
            stub = {"video_path": rec["video_path"]}
            tx = transcript_text(stub, max_chars=args.transcript_max_chars)
        else:
            tx = ""
        prompt = build_prompt(rec["question"], tx)
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
            "v2_attempt_used": rec.get("v2_attempt_used"),
        })

    if args.out_val and args.val_frac > 0:
        n_val = max(1, int(len(out_records) * args.val_frac))
        val, train = out_records[:n_val], out_records[n_val:]
    else:
        val, train = [], out_records

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    json.dump(train, open(args.out_train, "w"), indent=2)
    print(f"Wrote {len(train)} train pairs -> {args.out_train}")
    if val:
        json.dump(val, open(args.out_val, "w"), indent=2)
        print(f"Wrote {len(val)} val pairs   -> {args.out_val}")

    print("\n=== v2 assemble summary ===")
    print(f"  duplicates   dropped : {drop_dup}")
    print(f"  saturated    dropped : {drop_sat}  "
          f"(thresh={args.max_margin_per_tok})")
    print(f"  no-video     dropped : {drop_no_video}")
    print(f"  with_transcript      : {args.with_transcript}")
    by_ax_n: Counter = Counter(r["axis"] for r in out_records)
    for ax, n in sorted(by_ax_n.items()):
        print(f"  axis {ax:8s}        : {n}")


if __name__ == "__main__":
    main()
