"""
prepare_tsv.py — Convert video_meta.jsonl → VLMEvalKit-compatible TSV.

VLMEvalKit video MCQ datasets expect a TSV with columns:
  index | video | video_path | question | A | B | C | D | answer | question_type | time_reference

This script parses the embedded "(A) ... (B) ..." options out of the
question text and writes a clean TSV that VLMEvalKit can consume.

Usage:
    python prepare_tsv.py
"""

import json
import os
import re
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
META_JSONL = os.path.join(SCRIPT_DIR, "..", "video_meta.jsonl")
VIDEO_DIR = "/data/Pupil/lvbench_v2"
OUTPUT_TSV = os.path.join(SCRIPT_DIR, "LVBench_v2_MCQ.tsv")


def parse_question_options(raw: str):
    """
    Split "What year ...?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633"
    into (question_text, {A: "1636", B: "1366", C: "1363", D: "1633"}).
    """
    # Split on (A), (B), (C), (D), (E)
    parts = re.split(r"\n\(([A-E])\)\s*", raw)
    question_text = parts[0].strip()
    options = {}
    for i in range(1, len(parts), 2):
        letter = parts[i]
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        options[letter] = text
    return question_text, options


def main():
    with open(META_JSONL, "r") as f:
        videos = [json.loads(line) for line in f if line.strip()]

    rows = []
    idx = 0
    missing_videos = []

    for vid in videos:
        key = vid["key"]
        video_file = f"{key}_clean.mp4"
        video_path = os.path.join(VIDEO_DIR, video_file)

        if not os.path.exists(video_path):
            missing_videos.append(video_file)
            continue

        for qa in vid["qa"]:
            question_text, options = parse_question_options(qa["question"])
            row = {
                "index": idx,
                "video": key,
                "video_path": video_path,  # absolute path to video
                "question": question_text,
                "A": options.get("A", ""),
                "B": options.get("B", ""),
                "C": options.get("C", ""),
                "D": options.get("D", ""),
                "answer": qa["answer"],
                "question_type": json.dumps(qa.get("question_type", [])),
                "time_reference": qa.get("time_reference", ""),
                "uid": qa["uid"],
            }
            rows.append(row)
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)

    print(f"✅  Wrote {len(df)} QA items to {OUTPUT_TSV}")
    print(f"   Videos: {len(videos)} in meta, {len(videos) - len(missing_videos)} with files")
    if missing_videos:
        print(f"   ⚠  {len(missing_videos)} missing video files: {missing_videos[:5]}...")


if __name__ == "__main__":
    main()
