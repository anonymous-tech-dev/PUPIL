#!/usr/bin/env python3
"""
06_rewrite_video_paths.py
-------------------------
Produce a new pair of DPO json files where each row's `video` field points at
the pre-cut clip in <clips-dir>/<id>.mp4 instead of the full source video.

Originals are NOT modified. Rows whose clip is missing/failed are dropped and
listed in <out-dir>/DROPPED.<split>.txt so you can see what fell out.

Usage:
    python3 06_rewrite_video_paths.py \
        --in-train data_with_timestamps_v1/sof_dpo_train.json \
        --in-val   data_with_timestamps_v1/sof_dpo_train.val.json \
        --clips-dir /data/Pupil/clips_v1 \
        --out-dir   data_with_clips_v1
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def rewrite(rows: list, clips_dir: Path) -> tuple[list, list]:
    kept, dropped = [], []
    for r in rows:
        clip = clips_dir / f"{r['id']}.mp4"
        if not clip.exists() or clip.stat().st_size == 0:
            dropped.append(r["id"])
            continue
        new = dict(r)
        new["video_original"] = r["video"]
        new["video"] = str(clip)
        kept.append(new)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-train",  required=True, type=Path)
    ap.add_argument("--in-val",    required=True, type=Path)
    ap.add_argument("--clips-dir", required=True, type=Path)
    ap.add_argument("--out-dir",   required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, src in (("train", args.in_train), ("val", args.in_val)):
        rows = json.load(src.open())
        kept, dropped = rewrite(rows, args.clips_dir)
        dst = args.out_dir / src.name
        json.dump(kept, dst.open("w"), indent=2, ensure_ascii=False)
        if dropped:
            (args.out_dir / f"DROPPED.{split}.txt").write_text("\n".join(dropped))
        print(f"[{split}] kept {len(kept)}/{len(rows)}  dropped={len(dropped)}  -> {dst}")

    (args.out_dir / "PROVENANCE.json").write_text(json.dumps({
        "source_train": str(args.in_train),
        "source_val":   str(args.in_val),
        "clips_dir":    str(args.clips_dir),
        "rewrote":      "video -> <clips_dir>/<id>.mp4 (kept old as video_original)",
    }, indent=2))


if __name__ == "__main__":
    main()
