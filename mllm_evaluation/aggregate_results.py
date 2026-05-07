#!/usr/bin/env python3
"""Aggregate shard results into a single unified output directory."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def aggregate(model_name, num_shards, output_folder=None, model_dir_name=None):
    if output_folder is None:
        output_folder = config.OUTPUT_FOLDER
    # Allow ablations to override the per-model output directory name (mirrors
    # the MODEL_DIR_NAME knob in script_parallel.py).
    if not model_dir_name:
        model_dir_name = os.environ.get("MODEL_DIR_NAME", "").strip() or model_name
    base = os.path.join(config.OUTPUT_DIR, model_dir_name)
    merged_dir = os.path.join(base, output_folder)
    os.makedirs(merged_dir, exist_ok=True)
    shard_dirs = [os.path.join(base, f"{output_folder}_shard{i}") for i in range(num_shards) if os.path.isdir(os.path.join(base, f"{output_folder}_shard{i}"))]
    if not shard_dirs:
        print("❌ No shard directories found!"); return
    print(f"📂 Merging {len(shard_dirs)} shard dirs into {merged_dir}")
    all_files = set()
    for sd in shard_dirs:
        for f in os.listdir(sd):
            if f.endswith("_results.json"): all_files.add(f)
    total = 0
    for fname in sorted(all_files):
        seen_ids, merged_data = set(), []
        merged_path = os.path.join(merged_dir, fname)
        if os.path.exists(merged_path):
            try:
                for item in json.load(open(merged_path)):
                    qid = item.get("query_id", "")
                    qtxt = (item.get("question") or "").strip()
                    key = (qid, qtxt)
                    if key not in seen_ids: seen_ids.add(key); merged_data.append(item)
            except: pass
        for sd in shard_dirs:
            sf = os.path.join(sd, fname)
            if not os.path.exists(sf): continue
            try:
                for item in json.load(open(sf)):
                    qid = item.get("query_id", "")
                    qtxt = (item.get("question") or "").strip()
                    key = (qid, qtxt)
                    if key not in seen_ids: seen_ids.add(key); merged_data.append(item)
            except: pass
        json.dump(merged_data, open(merged_path, "w"), indent=4)
        total += len(merged_data)
    print(f"✅ Merged {total} query results across {len(all_files)} video files → {merged_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--output-folder", default=None)
    p.add_argument("--model-dir-name", default=None,
                   help="Override the per-model directory name "
                        "(used by ablation scripts). Defaults to --model or "
                        "the MODEL_DIR_NAME env var.")
    args = p.parse_args()
    aggregate(args.model, args.num_shards, args.output_folder, args.model_dir_name)
