#!/usr/bin/env python3
"""Rewrite cgbench_dpo_full.json to point at full-length train_vids.

For each pair, look up the qid in cgbench.json to get video_uid, then
rewrite path /clue_vids/{qid}.mp4 -> /train_vids/{video_uid}.mp4.
Drop pairs whose target video file does not exist on disk.

Also annotates each record with `_video_uid`, `_duration_sec`, `_sub_category`,
`_domain` so the dataset/trainer can filter / report per-category stats.
"""
import argparse, json, os, sys
from collections import Counter

CG_META = "/workspace/Pupil/contrastive_experiments/cgbench_setup/cgbench.json"
DEFAULT_FULL_DIR = "/data/Pupil/CGBench/train_vids"
TEST_JSON = "/workspace/Pupil/contrastive_experiments/final_sft_data/test.json"
VAL_JSON  = "/workspace/Pupil/contrastive_experiments/final_sft_data/val.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full_video_dir", default=DEFAULT_FULL_DIR)
    ap.add_argument("--max_duration_sec", type=float, default=None,
                    help="Drop pairs whose source video is longer than this.")
    ap.add_argument("--no_exclude_eval", action="store_true",
                    help="Disable test/val qid exclusion (NOT recommended).")
    args = ap.parse_args()

    # --- CRITICAL: build test/val qid blocklist to prevent leakage ---
    blocklist = set()
    if not args.no_exclude_eval:
        for split_path in (TEST_JSON, VAL_JSON):
            ids = {x["id"] for x in json.load(open(split_path))}
            blocklist |= ids
        print(f"Blocklist: {len(blocklist)} test+val qids will be excluded")

    print(f"Loading CGBench metadata: {CG_META}")
    cg = json.load(open(CG_META))
    qid_to = {str(r["qid"]): r for r in cg}

    print(f"Loading DPO pairs: {args.inp}")
    pairs = json.load(open(args.inp))
    print(f"  {len(pairs)} pairs")

    out = []
    drop_reason = Counter()
    for r in pairs:
        if r["id"] in blocklist:
            drop_reason["in_test_or_val"] += 1
            continue
        # id format: "cgbench_<qid>"
        qid = r["id"].split("_", 1)[-1]
        meta = qid_to.get(qid)
        if meta is None:
            drop_reason["no_cgbench_meta"] += 1
            continue
        uid = meta.get("video_uid")
        if not uid:
            drop_reason["no_video_uid"] += 1
            continue
        full_path = os.path.join(args.full_video_dir, f"{uid}.mp4")
        if not os.path.exists(full_path):
            drop_reason["video_file_missing"] += 1
            continue
        dur = float(meta.get("duration", 0))
        if args.max_duration_sec and dur > args.max_duration_sec:
            drop_reason["too_long"] += 1
            continue
        new = dict(r)
        new["video"] = full_path
        new["_video_uid"] = uid
        new["_duration_sec"] = dur
        new["_sub_category"] = meta.get("sub_category", "")
        new["_domain"] = meta.get("domain", "")
        out.append(new)

    print(f"\n  kept:    {len(out)}")
    for k, v in drop_reason.most_common():
        print(f"  drop {k:24s} {v}")

    # Duration histogram
    if out:
        durs = sorted([r["_duration_sec"] for r in out])
        print(f"\n  duration percentiles (sec):")
        for p in (10, 25, 50, 75, 90, 95, 99, 100):
            idx = min(len(durs)-1, int(len(durs)*p/100))
            print(f"    p{p:3d}  {durs[idx]:7.1f}s  ({durs[idx]/60:.1f} min)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {len(out)} pairs -> {args.out}")

    # Also write a val split ourselves to keep the API consistent.
    val_path = args.out.replace(".json", ".val.json")
    val_n = max(8, min(64, len(out)//100))
    val = out[:val_n]
    with open(val_path, "w") as f:
        json.dump(val, f)
    print(f"Wrote {val_n} val pairs -> {val_path}")

if __name__ == "__main__":
    main()
