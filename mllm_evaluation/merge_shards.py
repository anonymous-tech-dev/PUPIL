#!/usr/bin/env python3
"""
Merge shard result files into a single unified output directory.

Each shard writes to:  results/<model>/<output_folder>_shard<N>/<video>_results.json
This script merges them into:  results/<model>/<output_folder>/<video>_results.json

Deduplication is by (query_id, question) pair — safe to re-run.

Usage:
    python merge_shards.py --model qwen3_vl --num-shards 8
    python merge_shards.py --model qwen3_vl --num-shards 8 --output-folder final_1k_benchmark
"""

import os
import sys
import json
import argparse
import glob
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def merge(model_name, num_shards, output_folder=None, delete_shards=True):
    if output_folder is None:
        output_folder = config.OUTPUT_FOLDER
        # For finetuned models, mirror the suffix logic in script_parallel.py
        if model_name == "qwen3_vl_ft":
            adapter_tag = os.environ.get(
                "ADAPTER_TAG",
                os.path.basename(os.environ.get("ADAPTER_DIR", "ft").rstrip("/")),
            )
            output_folder = f"{output_folder}_ft_{adapter_tag}"

    base_dir = os.path.join(config.OUTPUT_DIR, model_name)
    merged_dir = os.path.join(base_dir, output_folder)
    os.makedirs(merged_dir, exist_ok=True)

    shard_dirs = []
    for s in range(num_shards):
        d = os.path.join(base_dir, f"{output_folder}_shard{s}")
        if os.path.isdir(d):
            shard_dirs.append(d)
        else:
            print(f"⚠️  Shard dir not found (may have had 0 work): {d}")

    if not shard_dirs:
        print("❌ No shard directories found. Nothing to merge.")
        return

    # Collect all result files across shards
    all_files = {}  # basename -> list of (shard_dir, filepath)
    for sd in shard_dirs:
        for fp in glob.glob(os.path.join(sd, "*_results.json")):
            basename = os.path.basename(fp)
            all_files.setdefault(basename, []).append(fp)

    total_queries = 0
    total_files = 0

    for basename, filepaths in sorted(all_files.items()):
        merged_path = os.path.join(merged_dir, basename)

        # Load existing merged file (for re-run safety)
        existing = []
        if os.path.exists(merged_path):
            try:
                with open(merged_path, "r") as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

        # Deduplicate by (query_id, question)
        seen = set()
        merged = []
        for item in existing:
            key = (item.get("query_id", ""), item.get("question", ""))
            if key not in seen:
                seen.add(key)
                merged.append(item)

        # Add from each shard
        for fp in filepaths:
            try:
                with open(fp, "r") as f:
                    shard_data = json.load(f)
                for item in shard_data:
                    key = (item.get("query_id", ""), item.get("question", ""))
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)
            except (json.JSONDecodeError, Exception) as e:
                print(f"  ⚠️ Error reading {fp}: {e}")

        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=4)

        total_queries += len(merged)
        total_files += 1

    print(f"\n{'='*60}")
    print(f"  MERGE COMPLETE — {model_name}")
    print(f"  Video files: {total_files}")
    print(f"  Total queries: {total_queries}")
    print(f"  Merged to: {merged_dir}")
    print(f"{'='*60}\n")

    # ── Verification & shard cleanup ──────────────────────────────
    if not delete_shards:
        return True

    # Load the query file to get the expected counts
    query_file = os.environ.get("EVAL_QUERY", config.QUERY_FILE_PATH)
    try:
        with open(query_file, "r") as f:
            query_data = json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load query file for verification ({query_file}): {e}")
        print("   Skipping shard deletion for safety.")
        return False

    expected_videos = {os.path.splitext(k)[0] for k in query_data.keys()}
    expected_total_queries = sum(len(qs) for qs in query_data.values())

    # Count what we actually have in merged dir
    merged_videos = set()
    actual_total_queries = 0
    for fp in glob.glob(os.path.join(merged_dir, "*_results.json")):
        basename = os.path.basename(fp)
        video_key = basename.replace("_results.json", "")
        merged_videos.add(video_key)
        try:
            with open(fp, "r") as f:
                actual_total_queries += len(json.load(f))
        except Exception:
            pass

    missing_videos = expected_videos - merged_videos
    video_ok = len(missing_videos) == 0
    query_ok = actual_total_queries >= expected_total_queries

    print(f"  🔍 Verification:")
    print(f"     Expected videos : {len(expected_videos)}  |  Merged videos : {len(merged_videos)}")
    print(f"     Expected queries: {expected_total_queries}  |  Merged queries: {actual_total_queries}")

    if not video_ok:
        print(f"  ❌ Missing {len(missing_videos)} video(s): {sorted(missing_videos)[:10]}...")
    if not query_ok:
        print(f"  ❌ Query count mismatch ({actual_total_queries} < {expected_total_queries})")

    if video_ok and query_ok:
        print("  ✅ Verification passed — deleting shard directories...")
        import shutil
        for sd in shard_dirs:
            shutil.rmtree(sd)
            print(f"     🗑  Deleted {sd}")
        print("  Done.\n")
        return True
    else:
        print("  ⚠️  Verification FAILED — shard directories kept for safety.\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge shard results")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--no-delete", action="store_true",
                        help="Skip verification & shard deletion (old behaviour)")
    args = parser.parse_args()
    merge(args.model, args.num_shards, args.output_folder,
          delete_shards=not args.no_delete)
