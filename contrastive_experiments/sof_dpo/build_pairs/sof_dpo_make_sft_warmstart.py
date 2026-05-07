"""
sof_dpo_make_sft_warmstart.py — Convert the assembled DPO json into the SFT
format used by src/train/old/train_sft.py (chosen-only, conversations schema).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dpo-json", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", default=None)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = json.load(open(args.in_dpo_json))
    rnd = random.Random(args.seed)
    rnd.shuffle(pairs)

    sft = []
    for p in pairs:
        prompt = p["prompt"]
        if "<video>" not in prompt:
            prompt = "<video>\n" + prompt
        sft.append({
            "id": p["id"],
            "video": p["video"],
            "source": "edubench-train-sft-warmstart",
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": p["chosen"]},
            ],
            "axis": p["axis"],
            "cognitive_category": p.get("cognitive_category", ""),
        })

    if args.out_val and args.val_frac > 0:
        n = max(1, int(len(sft) * args.val_frac))
        val, train = sft[:n], sft[n:]
    else:
        val, train = [], sft

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    json.dump(train, open(args.out_train, "w"), indent=2)
    print(f"wrote {len(train)} -> {args.out_train}")
    if val:
        json.dump(val, open(args.out_val, "w"), indent=2)
        print(f"wrote {len(val)}   -> {args.out_val}")


if __name__ == "__main__":
    main()
