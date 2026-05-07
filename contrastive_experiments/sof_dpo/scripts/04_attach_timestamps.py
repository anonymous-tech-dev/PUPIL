#!/usr/bin/env python3
"""
04_attach_timestamps.py
-----------------------
Re-emit the SoF-DPO train/val JSON files with a `timestamp_segments` field
joined from the canonical training queries jsonl.

Inputs (read-only):
  --in-train   sof_dpo_train.json
  --in-val     sof_dpo_train.val.json
  --queries    all_Pupil_train_queries_v0.jsonl  (has query_id + timestamp_segments)

Outputs (NEW folder, never overwriting the source files):
  <out-dir>/sof_dpo_train.json
  <out-dir>/sof_dpo_train.val.json
  <out-dir>/MISSING_TIMESTAMPS.txt   (any rows whose id was not in the queries db)

Join key: dpo_row["id"] == queries_row["query_id"]   (verified 2271/2271 + 121/121).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path


def load_queries_index(path: Path) -> dict:
    idx = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("query_id")
            if qid is None:
                continue
            idx[qid] = {
                "timestamp_segments": row.get("timestamp_segments", []),
                "source_folder": row.get("source_folder"),
                "annotations": row.get("annotations", {}),
            }
    return idx


def attach(rows: list, idx: dict) -> tuple[list, list]:
    out, missing = [], []
    for r in rows:
        qid = r["id"]
        meta = idx.get(qid)
        if meta is None:
            missing.append(qid)
            new = dict(r)
            new["timestamp_segments"] = []
        else:
            new = dict(r)
            new["timestamp_segments"] = meta["timestamp_segments"]
            # also surface source_folder for the clip-cutter — handy, non-breaking.
            if meta.get("source_folder") and "source_folder" not in new:
                new["source_folder"] = meta["source_folder"]
        out.append(new)
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-train", required=True, type=Path)
    ap.add_argument("--in-val",   required=True, type=Path)
    ap.add_argument("--queries",  required=True, type=Path)
    ap.add_argument("--out-dir",  required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] queries  : {args.queries}")
    idx = load_queries_index(args.queries)
    print(f"[load]   -> {len(idx)} query_ids indexed")

    for split, src in (("train", args.in_train), ("val", args.in_val)):
        print(f"[load] {split:5s}   : {src}")
        rows = json.load(src.open())
        out_rows, missing = attach(rows, idx)
        dst = args.out_dir / src.name
        with dst.open("w") as f:
            json.dump(out_rows, f, indent=2, ensure_ascii=False)
        n_with = sum(1 for r in out_rows if r["timestamp_segments"])
        print(f"[write] {split:5s}  : {dst}  ({n_with}/{len(out_rows)} have timestamps, {len(missing)} missing)")
        if missing:
            (args.out_dir / f"MISSING_TIMESTAMPS.{split}.txt").write_text("\n".join(missing))

    # also drop a tiny manifest so nobody is confused later about provenance.
    manifest = {
        "source_train": str(args.in_train),
        "source_val":   str(args.in_val),
        "queries_db":   str(args.queries),
        "join_key":     "dpo.id == queries.query_id",
        "added_fields": ["timestamp_segments", "source_folder"],
    }
    (args.out_dir / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] new data dir: {args.out_dir}")


if __name__ == "__main__":
    main()
