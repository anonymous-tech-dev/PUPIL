"""
Bidirectional consolidation script for the final_1k query dataset.

consolidate():
    Reads all *_clean_queries.json files from the 4 sof_* folders,
    strips metadata, and produces a single consolidated JSON keyed by
    video filename (e.g. "video_name_clean.mp4") → list of all 12 queries.

deconsolidate():
    Takes the consolidated JSON and recreates the exact original folder
    structure: sof_audio/, sof_visual/, sof_time/, sof_priority/ with
    per-video *_clean_queries.json files (3 queries each), keyed by the
    original full video path.

Usage:
    python consolidate.py consolidate   [--input-dir DIR] [--output FILE]
    python consolidate.py deconsolidate [--input FILE]    [--output-dir DIR]
"""

import argparse
import json
import os
import glob
from collections import defaultdict

# ── constants ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDERS = ["sof_audio", "sof_priority", "sof_time", "sof_visual"]
VIDEO_PATH_PREFIX = (
    "/home/Pupil/dataset_curation/dataset/videos_db/final_1k/"
)
DEFAULT_CONSOLIDATED = os.path.join(BASE_DIR, "final_consolidated_1k.json")


# ── consolidate ────────────────────────────────────────────────────────
def consolidate(input_dir: str, output_file: str):
    """Merge all per-folder query JSONs into one consolidated file."""
    all_queries: dict[str, list] = defaultdict(list)
    file_count = 0

    for folder in FOLDERS:
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"  ⚠  Skipping missing folder: {folder_path}")
            continue

        query_files = sorted(glob.glob(os.path.join(folder_path, "*_clean_queries.json")))
        for qf in query_files:
            with open(qf, "r", encoding="utf-8") as f:
                data = json.load(f)

            for video_path, queries in data.items():
                # key by just the filename, e.g. "3_perplexing_physics_problems_clean.mp4"
                video_name = os.path.basename(video_path)
                all_queries[video_name].extend(queries)
            file_count += 1

    # sort by video name for deterministic output
    consolidated = {k: all_queries[k] for k in sorted(all_queries)}

    total_q = sum(len(v) for v in consolidated.values())
    print(f"  ✓  Read {file_count} query files across {len(FOLDERS)} folders")
    print(f"  ✓  {len(consolidated)} videos, {total_q} total queries")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    print(f"  ✓  Written to {output_file}")


# ── deconsolidate ──────────────────────────────────────────────────────
def deconsolidate(input_file: str, output_dir: str):
    """Split a consolidated JSON back into the original sof_* folder structure."""
    with open(input_file, "r", encoding="utf-8") as f:
        consolidated = json.load(f)

    # pipeline_mode → folder name mapping
    mode_to_folder = {
        "audio": "sof_audio",
        "priority": "sof_priority",
        "time": "sof_time",
        "visual": "sof_visual",
    }

    # bucket queries by (video_name, pipeline_mode)
    # structure: { folder_name: { video_name: [queries] } }
    buckets: dict[str, dict[str, list]] = {f: defaultdict(list) for f in FOLDERS}

    for video_name, queries in consolidated.items():
        for q in queries:
            mode = q.get("annotations", {}).get("pipeline_mode", "")
            folder = mode_to_folder.get(mode)
            if folder is None:
                print(f"  ⚠  Unknown pipeline_mode '{mode}' in query {q.get('query_id')}, skipping")
                continue
            buckets[folder][video_name].append(q)

    # write back to original file structure
    files_written = 0
    for folder_name in FOLDERS:
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for video_name, queries in sorted(buckets[folder_name].items()):
            # derive the original filename: "video_name_clean.mp4" → "video_name_clean_queries.json"
            stem = os.path.splitext(video_name)[0]  # "video_name_clean"
            out_filename = f"{stem}_queries.json"
            out_path = os.path.join(folder_path, out_filename)

            # reconstruct the original full video path key
            full_video_path = VIDEO_PATH_PREFIX + video_name

            file_data = {full_video_path: queries}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(file_data, f, indent=2, ensure_ascii=False)
            files_written += 1

    total_q = sum(
        len(q) for vids in buckets.values() for q in vids.values()
    )
    print(f"  ✓  Wrote {files_written} query files into {output_dir}")
    print(f"  ✓  {total_q} total queries across {len(FOLDERS)} folders")


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Consolidate / deconsolidate final_1k query dataset"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # consolidate
    p_con = sub.add_parser("consolidate", help="Merge sof_* folders → single JSON")
    p_con.add_argument(
        "--input-dir", default=BASE_DIR,
        help=f"Root dir containing sof_* folders (default: {BASE_DIR})"
    )
    p_con.add_argument(
        "--output", default=DEFAULT_CONSOLIDATED,
        help=f"Output consolidated JSON path (default: {DEFAULT_CONSOLIDATED})"
    )

    # deconsolidate
    p_dec = sub.add_parser("deconsolidate", help="Split consolidated JSON → sof_* folders")
    p_dec.add_argument(
        "--input", default=DEFAULT_CONSOLIDATED,
        help=f"Input consolidated JSON (default: {DEFAULT_CONSOLIDATED})"
    )
    p_dec.add_argument(
        "--output-dir", default=BASE_DIR,
        help=f"Output root dir for sof_* folders (default: {BASE_DIR})"
    )

    args = parser.parse_args()

    if args.command == "consolidate":
        print("Consolidating…")
        consolidate(args.input_dir, args.output)
    elif args.command == "deconsolidate":
        print("Deconsolidating…")
        deconsolidate(args.input, args.output_dir)


if __name__ == "__main__":
    main()
