#!/usr/bin/env python3
"""
02_prepare_lvbench_tsv.py
=========================
Builds the LVBench annotation TSV required by VLMEvalKit from:
  1. Your local video_meta.jsonl  (tailored to available videos)
  2. Your local video files at VIDEO_ROOT

Output: $LMUData/LVBench/LVBench.tsv
        $LMUData/LVBench/videos/  -> symlink to VIDEO_ROOT

video_meta.jsonl format (per line):
  {
    "key": "<youtube_id>",
    "type": "<genre>",
    "qa": [
      {
        "uid": "55",
        "question": "What year ...?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633",
        "answer": "D",
        "question_type": ["key information retrieval"],
        "time_reference": "00:15-00:19"
      }, ...
    ],
    "video_info": { "duration_minutes": ..., "fps": ..., "resolution": {...} }
  }

References:
  • VLMEvalKit TSV schema (Development.md):
      https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md
  • VLMEvalKit LMUData convention:
      https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md
"""

import os
import os.path as osp
import re
import json
import argparse
import pandas as pd
from pathlib import Path


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_META_JSONL = "/workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/video_meta.jsonl"
DEFAULT_VIDEO_ROOT = "/data/Pupil/lvbench_v2"
DEFAULT_LMU_ROOT   = os.environ.get("LMUData", osp.expanduser("~/LMUData"))

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

# Regex to split "(A) ... (B) ... (C) ... (D) ..." from the question text
_OPT_RE = re.compile(
    r"\n\(A\)\s*(.*?)\s*\n\(B\)\s*(.*?)\s*\n\(C\)\s*(.*?)\s*\n\(D\)\s*(.*?)\s*$",
    re.DOTALL,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build LVBench TSV for VLMEvalKit")
    p.add_argument(
        "--meta_jsonl", default=DEFAULT_META_JSONL,
        help="Path to video_meta.jsonl (default: %(default)s)"
    )
    p.add_argument(
        "--video_root", default=DEFAULT_VIDEO_ROOT,
        help="Directory containing your local LVBench video files "
             "(default: %(default)s)"
    )
    p.add_argument(
        "--lmu_root", default=DEFAULT_LMU_ROOT,
        help="VLMEvalKit LMUData root directory (default: %(default)s). "
             "Override with env var LMUData."
    )
    p.add_argument(
        "--output_tsv", default=None,
        help="Override output TSV path (default: $LMUData/LVBench/LVBench.tsv)"
    )
    return p.parse_args()


def build_video_index(video_root: str) -> dict:
    """
    Scan VIDEO_ROOT and build a mapping  video_stem -> full_path.
    LVBench videos are identified by YouTube ID (the filename stem).
    """
    index = {}
    root = Path(video_root)
    if not root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    for f in root.iterdir():
        if f.suffix.lower() in VIDEO_EXTS:
            index[f.stem] = str(f)
            # LVBench v2 videos have a "_clean" suffix (e.g. Cm73ma6Ibcs_clean.mp4)
            # but annotations use the bare YouTube ID (Cm73ma6Ibcs).
            bare = f.stem.removesuffix("_clean")
            if bare != f.stem:
                index[bare] = str(f)

    print(f"  Found {len(index)} video entries in {video_root}")
    return index


def parse_question_options(raw_question: str) -> tuple[str, list[str]]:
    """
    Split a question string like:
        "What year ...?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633"
    into (question_text, [opt_A, opt_B, opt_C, opt_D]).
    """
    m = _OPT_RE.search(raw_question)
    if m:
        question_text = raw_question[: m.start()].strip()
        options = [m.group(i).strip() for i in range(1, 5)]
        return question_text, options
    # Fallback: return whole string as question, empty options
    return raw_question.strip(), []


def load_meta_jsonl(meta_path: str) -> list[dict]:
    """Load the local video_meta.jsonl and flatten into per-question rows."""
    rows = []
    with open(meta_path) as f:
        for line in f:
            entry = json.loads(line)
            video_key = entry["key"]
            video_type = entry.get("type", "Unknown")
            for qa in entry["qa"]:
                rows.append({
                    "video_key": video_key,
                    "video_type": video_type,
                    "uid": qa["uid"],
                    "raw_question": qa["question"],
                    "answer": qa["answer"],
                    "question_type": qa.get("question_type", []),
                    "time_reference": qa.get("time_reference", ""),
                })
    return rows


def build_tsv(rows: list[dict], video_index: dict) -> pd.DataFrame:
    """
    Convert flattened QA rows into VLMEvalKit TSV format.

    VLMEvalKit TSV schema for video MCQ:
        index | video | question | A | B | C | D | answer | category
    """
    records = []
    skipped_no_video = 0
    skipped_bad_parse = 0

    for i, row in enumerate(rows):
        vid_key = row["video_key"]

        if vid_key not in video_index:
            skipped_no_video += 1
            continue

        video_filename = osp.basename(video_index[vid_key])

        question_text, options = parse_question_options(row["raw_question"])

        if len(options) < 2:
            skipped_bad_parse += 1
            continue

        while len(options) < 4:
            options.append("")

        # question_type is a list like ["key information retrieval"]
        qt = row["question_type"]
        category = ", ".join(qt) if isinstance(qt, list) else str(qt)

        records.append({
            "index":    i,
            "video":    video_filename,
            "question": question_text,
            "A":        options[0],
            "B":        options[1],
            "C":        options[2],
            "D":        options[3],
            "answer":   row["answer"].strip().upper(),
            "category": category,
        })

    print(f"  Converted {len(records)} questions into TSV rows")
    if skipped_no_video:
        print(f"  ⚠  Skipped {skipped_no_video} questions — video not found locally")
    if skipped_bad_parse:
        print(f"  ⚠  Skipped {skipped_bad_parse} questions — could not parse options")

    return pd.DataFrame(records)


def main():
    args = parse_args()

    print("\n[1/3] Scanning local video directory ...")
    video_index = build_video_index(args.video_root)

    print("\n[2/3] Loading annotations from local JSONL ...")
    print(f"  Reading {args.meta_jsonl} ...")
    rows = load_meta_jsonl(args.meta_jsonl)
    print(f"  Loaded {len(rows)} QA pairs from {len(set(r['video_key'] for r in rows))} videos")

    print("\n[3/3] Building TSV ...")
    df = build_tsv(rows, video_index)

    # ── Output paths ──────────────────────────────────────────────────────────
    out_dir = osp.join(args.lmu_root, "LVBench")
    os.makedirs(out_dir, exist_ok=True)

    # Symlink videos directory to local video root
    videos_link = osp.join(out_dir, "videos")
    if osp.islink(videos_link):
        os.remove(videos_link)
    if osp.isdir(videos_link):
        print(f"  ⚠  {videos_link} is an existing directory, not symlinking.")
        print(f"     Make sure it contains the video files.")
    else:
        os.symlink(args.video_root, videos_link)
        print(f"  Symlinked: {videos_link} → {args.video_root}")

    # Write TSV
    tsv_path = args.output_tsv or osp.join(out_dir, "LVBench.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\n  Wrote {len(df)} rows to {tsv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(df) > 0:
        print("\n── Category breakdown ──────────────────────────────────────────")
        for cat, grp in df.groupby("category"):
            print(f"  {cat:<35}  {len(grp):>4} questions")
        print(f"  {'TOTAL':<35}  {len(df):>4} questions")
        print(f"  Unique videos: {df['video'].nunique()}")

    print("\n── Next step ────────────────────────────────────────────────────")
    print("  bash run_qwen25vl.sh   # GPU 0")
    print("  bash run_qwen3vl.sh    # GPU 1")
    print("────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
