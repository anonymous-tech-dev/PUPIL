#!/usr/bin/env python3
"""
===============================================================================
SFT Data Formatting Script for Contrastive Learning Experiments
===============================================================================
Unifies three data sources (CGBench, EduBench, FineVideo) into a single
LLaVA-style JSON format suitable for the Qwen-VL-Series-Finetune pipeline.

Outputs:
  - train.json  : interleaved training data (blocks of same-source samples)
  - val.json    : validation data with source labels
  - test.json   : test data with source labels

Each sample follows the LLaVA conversation format:
{
  "id": "cgbench_26",
  "video": "/path/to/video.mp4",
  "source": "cgbench" | "edubench" | "finevideo",
  "conversations": [
    {"from": "human", "value": "<video>\nQuestion"},
    {"from": "gpt", "value": "Answer"}
  ],
  "metadata": {
    "timestamps_sec": [[5, 7], ...],   # ground-truth temporal segments
    "duration_sec": 1980,              # full video duration in seconds
    "has_timestamps": true,            # whether timestamps are available
    "original_id": "...",              # original ID from the source dataset
    "domain": "...",                   # domain/category label (if any)
    "sub_category": "..."             # sub-category label (if any)
  }
}

Usage:
  python sft_format_data.py \
    --cgbench_dir ./cgbench_dataset/sft_data/sft_start1 \
    --cgbench_reasoning_dir ./cgbench_dataset/sft_data/sft_start4 \
    --cgbench_video_dir /data/cgbench/videos \
    --edubench_dir ../dataset_curation/dataset/queries_db/final_1k \
    --edubench_video_dir /data/Pupil/dataset_curation/dataset/videos_db/final_1k \
    --finevideo_path ./longvila_setup/finevideo_sft_long.json \
    --finevideo_video_dir /data/Pupil/FineVid/vids \
    --output_dir ./dataset \
    --use_reasoning_traces \
    --seed 42
===============================================================================
"""

import argparse
import json
import glob
import os
import random
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple


# =============================================================================
# STAGE 1: Argument Parsing — all knobs for data sourcing and splitting
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Format CGBench + EduBench + FineVideo into unified SFT data"
    )

    # ── Data source paths ──────────────────────────────────────────────────
    parser.add_argument(
        "--cgbench_dir",
        type=str,
        default="/data/Pupil/CGBench/sft_data/sft_start1",
        help="Directory containing CGBench SFT JSONs (without reasoning traces)"
    )
    parser.add_argument(
        "--cgbench_reasoning_dir",
        type=str,
        default="/data/Pupil/CGBench/sft_data/sft_start4",
        help="Directory containing CGBench SFT JSONs WITH reasoning traces"
    )
    parser.add_argument(
        "--cgbench_video_dir",
        type=str,
        default="/data/Pupil/CGBench/clue_vids",
        help="Root directory where CGBench videos are stored (files: {video_uid}.mp4)"
    )
    parser.add_argument(
        "--edubench_dir",
        type=str,
        default="../dataset_curation/dataset/queries_db/final_1k",
        help="Root directory containing EduBench query JSONs (with subdirs sof_visual, sof_audio, etc.)"
    )
    parser.add_argument(
        "--edubench_video_dir",
        type=str,
        default="/data/Pupil/dataset_curation/dataset/videos_db/final_1k",
        help="Directory where EduBench videos actually reside (for path remapping)"
    )
    parser.add_argument(
        "--finevideo_path",
        type=str,
        default="/workspace/Pupil/contrastive_experiments/finevid_setup/finevideo_sft_long.json",
        help="Path to FineVideo SFT JSON (already in LLaVA format)"
    )
    parser.add_argument(
        "--finevideo_video_dir",
        type=str,
        default="/data/Pupil/FineVid/vids_v2",
        help="Directory where FineVideo videos reside (for path remapping)"
    )

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/Pupil/contrastive_experiments/final_sft_data",
        help="Directory to write train.json, val.json, test.json"
    )

    # ── Split sizes (knobs) ────────────────────────────────────────────────
    parser.add_argument("--cgbench_train_n", type=int, default=10500,
                        help="Number of CGBench samples for training")
    parser.add_argument("--cgbench_test_n", type=int, default=750,
                        help="Number of CGBench samples for testing")
    parser.add_argument("--cgbench_val_n", type=int, default=750,
                        help="Number of CGBench samples for validation (remainder auto-computed if 0)")

    parser.add_argument("--edubench_train_n", type=int, default=0,
                        help="Number of EduBench samples for training")
    parser.add_argument("--edubench_test_n", type=int, default=0,
                        help="Number of EduBench samples for testing")
    parser.add_argument("--edubench_val_n", type=int, default=0,
                        help="Number of EduBench samples for validation (remainder auto-computed if 0)")

    parser.add_argument("--finevideo_train_n", type=int, default=900,
                        help="Number of FineVideo VIDEOS for training (each explodes to ~7 QA pairs)")
    parser.add_argument("--finevideo_val_n", type=int, default=100,
                        help="Number of FineVideo VIDEOS for validation (each explodes to ~7 QA pairs)")
    parser.add_argument("--finevideo_test_n", type=int, default=0,
                        help="Number of FineVideo VIDEOS for testing (each explodes to ~7 QA pairs)")

    # ── Reasoning traces ───────────────────────────────────────────────────
    parser.add_argument(
        "--use_reasoning_traces",
        action="store_true",
        default=False,
        help="If set, use CGBench data with reasoning traces (sft_start4) and "
             "append reasoning_trace to the answer"
    )

    # ── Block interleaving ─────────────────────────────────────────────────
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Number of samples per block for interleaving. "
             "Should equal per_device_batch_size * num_devices."
    )

    # ── Misc ───────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


# =============================================================================
# STAGE 2: Data loading helpers — one per source
# =============================================================================

def load_cgbench(
    data_dir: str,
    reasoning_dir: str,
    video_dir: str,
    use_reasoning: bool
) -> List[Dict[str, Any]]:
    """
    Load CGBench data from all JSON shards in the given directory.
    If use_reasoning is True, loads from reasoning_dir (sft_start4) instead
    and appends the reasoning_trace to the better_answer.

    Returns a list of unified-format dicts.
    """
    src_dir = reasoning_dir if use_reasoning else data_dir
    json_files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    if not json_files:
        print(f"[WARN] No CGBench JSON files found in {src_dir}")
        return []

    raw_entries = []
    for fpath in json_files:
        with open(fpath, "r") as f:
            raw_entries.extend(json.load(f))

    # De-duplicate by qid (shards may overlap)
    seen_qids = set()
    deduped = []
    for entry in raw_entries:
        if entry["qid"] not in seen_qids:
            seen_qids.add(entry["qid"])
            deduped.append(entry)
    raw_entries = deduped

    print(f"[INFO] Loaded {len(raw_entries)} CGBench entries (use_reasoning={use_reasoning})")

    unified = []
    for entry in raw_entries:
        # Videos are named by qid (e.g. 26.mp4), not video_uid
        video_path = os.path.join(video_dir, f"{entry['qid']}.mp4")

        # Build the answer: use better_answer, optionally prepend reasoning trace
        answer = entry.get("better_answer", entry["answer"])
        if use_reasoning and "reasoning_trace" in entry and entry["reasoning_trace"]:
            answer = f"Reasoning: {entry['reasoning_trace']}\n\nAnswer: {answer}"

        # Parse clue_intervals → timestamps_sec (list of [start, end] in seconds)
        timestamps_sec = []
        for interval in entry.get("clue_intervals", []):
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                timestamps_sec.append([float(interval[0]), float(interval[1])])

        unified.append({
            "id": f"cgbench_{entry['qid']}",
            "video": video_path,
            "source": "cgbench",
            "conversations": [
                {"from": "human", "value": f"<video>\n{entry['question']}"},
                {"from": "gpt", "value": answer},
            ],
            "metadata": {
                "timestamps_sec": timestamps_sec,
                "duration_sec": float(entry.get("duration", 0)),
                "has_timestamps": len(timestamps_sec) > 0,
                "original_id": str(entry["qid"]),
                "video_uid": entry["video_uid"],
                "domain": entry.get("domain", ""),
                "sub_category": entry.get("sub_category", ""),
            },
        })

    return unified


def _parse_timestamp_hms(ts_str: str) -> float:
    """Convert 'HH:MM:SS' or 'MM:SS' to seconds."""
    parts = ts_str.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    else:
        return parts[0]


def load_edubench(
    data_dir: str,
    video_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load EduBench data from all subdirectories (sof_visual, sof_audio, etc.).
    Each JSON file maps a video path → list of queries.
    We remap the stored video paths to the actual video_dir.

    Returns a list of unified-format dicts.
    """
    json_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True))
    # Exclude any non-query files (like the xlsx)
    json_files = [f for f in json_files if f.endswith("_queries.json")]

    if not json_files:
        print(f"[WARN] No EduBench query JSON files found in {data_dir}")
        return []

    unified = []
    for fpath in json_files:
        with open(fpath, "r") as f:
            data = json.load(f)

        for original_video_path, queries in data.items():
            # Remap: extract just the filename and place it in video_dir
            video_filename = os.path.basename(original_video_path)
            video_path = os.path.join(video_dir, video_filename)

            for q in queries:
                # Parse timestamp_segments if present
                timestamps_sec = []
                for seg in q.get("timestamp_segments", []):
                    start_sec = _parse_timestamp_hms(seg["start"])
                    end_sec = _parse_timestamp_hms(seg["end"])
                    timestamps_sec.append([start_sec, end_sec])

                query_id = q.get("query_id", "")
                question = q.get("question", "")
                ground_truth = q.get("ground_truth", "")

                # Skip degenerate entries (some pipeline failures produce garbage)
                if not question or not ground_truth:
                    continue
                if len(question) > 500 and "I can't generate" in question:
                    continue

                annotations = q.get("annotations", {})

                unified.append({
                    "id": f"edubench_{query_id}",
                    "video": video_path,
                    "source": "edubench",
                    "conversations": [
                        {"from": "human", "value": f"<video>\n{question}"},
                        {"from": "gpt", "value": ground_truth},
                    ],
                    "metadata": {
                        "timestamps_sec": timestamps_sec,
                        "duration_sec": 0.0,  # Not available in EduBench; will be filled at load time
                        "has_timestamps": len(timestamps_sec) > 0,
                        "original_id": query_id,
                        "pipeline_mode": annotations.get("pipeline_mode", ""),
                        "cognitive_category": annotations.get("cognitive_category", ""),
                    },
                })

    print(f"[INFO] Loaded {len(unified)} EduBench entries")
    return unified


def load_finevideo(
    json_path: str,
    video_dir: str,
) -> List[Dict[str, Any]]:
    """
    Load FineVideo data.  Already in LLaVA multi-turn format.
    We remap video paths to the actual video_dir and add source metadata.

    Returns video-level entries (1 per video, multi-turn conversations intact).
    Use flatten_finevideo() after splitting to explode into individual QA pairs.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    if not raw:
        print(f"[WARN] FineVideo JSON is empty: {json_path}")
        return []

    unified = []
    for entry in raw:
        video_filename = os.path.basename(entry["video"])
        video_path = os.path.join(video_dir, video_filename)

        unified.append({
            "id": entry["id"],
            "video": video_path,
            "source": "finevideo",
            "conversations": entry["conversations"],  # Multi-turn, kept intact
            "metadata": {
                "timestamps_sec": [],
                "duration_sec": 0.0,
                "has_timestamps": False,
                "original_id": entry["id"],
            },
        })

    print(f"[INFO] Loaded {len(unified)} FineVideo video-level entries")
    return unified


def flatten_finevideo(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten multi-turn FineVideo entries into individual single-turn QA pairs.
    Each entry with N QA turns becomes N separate training samples, all sharing
    the same video path but with independent conversations.

    This must be called AFTER splitting by video to prevent train/test leakage.
    """
    flattened = []
    for entry in entries:
        convs = entry["conversations"]
        # Each pair of (human, gpt) turns becomes one entry
        for turn_idx in range(0, len(convs), 2):
            if turn_idx + 1 >= len(convs):
                break  # Incomplete pair at the end
            human_turn = convs[turn_idx]
            gpt_turn = convs[turn_idx + 1]
            if human_turn["from"] != "human" or gpt_turn["from"] != "gpt":
                continue  # Skip malformed pairs

            qa_id = turn_idx // 2

            # Ensure every human turn has the <video> token so data on
            # disk is consistent (the original multi-turn format only has
            # it in the first turn).
            human_value = human_turn["value"]
            if "<video>" not in human_value:
                human_value = "<video>\n" + human_value

            flattened.append({
                "id": f"{entry['id']}_qa{qa_id}",
                "video": entry["video"],
                "source": "finevideo",
                "conversations": [
                    {"from": "human", "value": human_value},
                    gpt_turn,
                ],
                "metadata": {
                    **entry["metadata"],
                    "original_id": entry["id"],
                    "qa_index": qa_id,
                },
            })

    print(f"[INFO] Flattened {len(entries)} FineVideo videos → {len(flattened)} QA pairs")
    return flattened


# =============================================================================
# STAGE 3: Train / Val / Test splitting with configurable sizes
# =============================================================================

def split_dataset(
    data: List[Dict[str, Any]],
    train_n: int,
    val_n: int,
    test_n: int,
    rng: random.Random,
    source_name: str,
) -> Tuple[List, List, List]:
    """
    Shuffle and split a single-source dataset into train/val/test.
    If val_n == 0, remainder after train+test goes to val.
    If there aren't enough samples, we take what we can.
    """
    rng.shuffle(data)
    total = len(data)

    # Clamp to available data
    test_n = min(test_n, total)
    remaining = total - test_n
    train_n = min(train_n, remaining)
    if val_n == 0:
        val_n = remaining - train_n
    else:
        val_n = min(val_n, remaining - train_n)

    test = data[:test_n]
    train = data[test_n : test_n + train_n]
    val = data[test_n + train_n : test_n + train_n + val_n]

    print(f"[INFO] {source_name} split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# =============================================================================
# STAGE 4: Block-interleaved training data generation
# =============================================================================

def interleave_training_data(
    sources: Dict[str, List[Dict[str, Any]]],
    block_size: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Interleave training data from multiple sources in blocks of block_size.

    Algorithm:
      1. For each block position, randomly select a source (uniform 1/N).
      2. Fill the block with consecutive samples from that source.
      3. If a source is exhausted, cycle (wrap around) for this epoch.
      4. Drop the last block if incomplete.

    This ensures:
      - No mixing of sources within a block (= within a batch)
      - Uniform exposure to all sources over training
      - Reproducibility via the seeded RNG
    """
    source_names = sorted(sources.keys())
    # Pre-shuffle each source
    for name in source_names:
        rng.shuffle(sources[name])

    # Track position (cursor) within each source for cycling
    cursors = {name: 0 for name in source_names}
    total_samples = sum(len(v) for v in sources.values())

    # Number of blocks we need to cover all data at least once
    # We produce enough blocks so each source is seen at least once fully
    max_source_len = max(len(v) for v in sources.values())
    # Each source gets ~1/N of the blocks, so total blocks ≈ N * max_source_len / block_size
    num_blocks = (len(source_names) * max_source_len) // block_size + 1

    interleaved = []
    source_block_counts = defaultdict(int)

    for _ in range(num_blocks):
        # Randomly pick a source (uniform 1-in-N)
        chosen = rng.choice(source_names)
        source_data = sources[chosen]
        if len(source_data) == 0:
            continue

        block = []
        for _ in range(block_size):
            idx = cursors[chosen] % len(source_data)  # Cycle if exhausted
            block.append(source_data[idx])
            cursors[chosen] += 1

        interleaved.extend(block)
        source_block_counts[chosen] += 1

    # Drop last incomplete block (if total isn't divisible by block_size)
    remainder = len(interleaved) % block_size
    if remainder > 0:
        interleaved = interleaved[:-remainder]

    print(f"[INFO] Interleaved {len(interleaved)} training samples in {len(interleaved)//block_size} blocks")
    for name, count in sorted(source_block_counts.items()):
        print(f"  {name}: {count} blocks ({count * block_size} samples)")

    return interleaved


# =============================================================================
# STAGE 5: Main — load, split, interleave, save
# =============================================================================

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load all sources ───────────────────────────────────────────────────
    cgbench_data = load_cgbench(
        data_dir=args.cgbench_dir,
        reasoning_dir=args.cgbench_reasoning_dir,
        video_dir=args.cgbench_video_dir,
        use_reasoning=args.use_reasoning_traces,
    )
    if args.edubench_train_n + args.edubench_val_n + args.edubench_test_n > 0:
        edubench_data = load_edubench(
            data_dir=args.edubench_dir,
            video_dir=args.edubench_video_dir,
        )
    else:
        print("[INFO] Skipping EduBench (all split sizes are 0)")
        edubench_data = []
    finevideo_data = load_finevideo(
        json_path=args.finevideo_path,
        video_dir=args.finevideo_video_dir,
    )

    # ── Split each source ──────────────────────────────────────────────────
    cg_train, cg_val, cg_test = split_dataset(
        cgbench_data, args.cgbench_train_n, args.cgbench_val_n, args.cgbench_test_n,
        rng, "CGBench"
    )
    edu_train, edu_val, edu_test = split_dataset(
        edubench_data, args.edubench_train_n, args.edubench_val_n, args.edubench_test_n,
        rng, "EduBench"
    )
    # FineVideo: split by VIDEO first (to prevent leakage), then flatten to QA pairs
    fv_train_vids, fv_val_vids, fv_test_vids = split_dataset(
        finevideo_data, args.finevideo_train_n, args.finevideo_val_n, args.finevideo_test_n,
        rng, "FineVideo (videos)"
    )
    fv_train = flatten_finevideo(fv_train_vids)
    fv_val = flatten_finevideo(fv_val_vids)
    fv_test = flatten_finevideo(fv_test_vids)

    # ── Interleave training data ───────────────────────────────────────────
    train_sources = {}
    if cg_train:
        train_sources["cgbench"] = cg_train
    if edu_train:
        train_sources["edubench"] = edu_train
    if fv_train:
        train_sources["finevideo"] = fv_train

    train_data = interleave_training_data(train_sources, args.block_size, rng)

    # ── Combine val and test (keep source labels for filtering) ────────────
    val_data = cg_val + edu_val + fv_val
    rng.shuffle(val_data)

    test_data = cg_test + edu_test + fv_test
    rng.shuffle(test_data)

    # ── Save ───────────────────────────────────────────────────────────────
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out_path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved {len(split_data)} samples to {out_path}")

    # ── Save config for reproducibility ────────────────────────────────────
    config_path = os.path.join(args.output_dir, "data_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[INFO] Saved data config to {config_path}")

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA FORMATTING COMPLETE")
    print("=" * 60)
    source_counts = defaultdict(int)
    for s in train_data:
        source_counts[s["source"]] += 1
    print(f"Training: {len(train_data)} total")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
