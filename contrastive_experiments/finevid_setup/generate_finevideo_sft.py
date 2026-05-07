#!/usr/bin/env python3
"""
===============================================================================
FineVideo Download + SFT Generator (Memory-Safe, Resumable)
===============================================================================
Streams the FineVideo HF dataset, downloads mp4 files for Education and
Science & Technology categories (≥ MIN_DURATION), and builds a LLaVA-style
SFT JSON file with QA conversations.

Features:
  - Streaming mode: one video at a time in RAM
  - Hot-resume: skips already-downloaded videos on restart
  - Aggressive saves: JSON is written after every new video

Usage:
  python generate_finevideo_sft.py --hf_token <YOUR_HF_TOKEN>

  # Resume after crash (just re-run the same command):
  python generate_finevideo_sft.py --hf_token <YOUR_HF_TOKEN>

  # Custom targets:
  python generate_finevideo_sft.py --hf_token <YOUR_HF_TOKEN> \
    --edu_target 584 --sci_target 416 --min_duration 540
===============================================================================
"""

import os
import json
import argparse
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Download FineVideo + generate SFT JSON")
    parser.add_argument("--hf_token", type=str, required=True,
                        help="HuggingFace API token")
    parser.add_argument("--hf_dataset", type=str, default="HuggingFaceFV/finevideo",
                        help="HuggingFace dataset ID")
    parser.add_argument("--vids_dir", type=str,
                        default="/data/Pupil/FineVid/vids_v2",
                        help="Directory to save downloaded mp4 files")
    parser.add_argument("--output_json", type=str,
                        default="/workspace/Pupil/contrastive_experiments/finevid_setup/finevideo_sft_long.json",
                        help="Path to write the SFT JSON")
    parser.add_argument("--edu_target", type=int, default=584,
                        help="Number of Education videos to collect")
    parser.add_argument("--sci_target", type=int, default=416,
                        help="Number of Science & Technology videos to collect")
    parser.add_argument("--min_duration", type=int, default=540,
                        help="Minimum video duration in seconds (default 540 = 9min)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    vids_dir = args.vids_dir
    output_json_path = args.output_json
    os.makedirs(vids_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)

    target_categories = ["Education", "Science & Technology"]
    TARGETS = {
        "Education": args.edu_target,
        "Science & Technology": args.sci_target,
    }
    MIN_DURATION_SECONDS = args.min_duration

    # ── Load existing progress (hot-resume) ────────────────────────────────
    sft_data = []
    existing_counts = {cat: 0 for cat in target_categories}

    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            sft_data = json.load(f)
        print(f"[RESUME] Loaded existing JSON with {len(sft_data)} entries.")

        for item in sft_data:
            if "edu_" in item["id"]:
                existing_counts["Education"] += 1
            elif "sci_" in item["id"]:
                existing_counts["Science & Technology"] += 1

    print(f"[INFO] Current progress -> Education: {existing_counts['Education']}/{TARGETS['Education']} "
          f"| Sci & Tech: {existing_counts['Science & Technology']}/{TARGETS['Science & Technology']}")

    if all(existing_counts[cat] >= TARGETS[cat] for cat in target_categories):
        print("[INFO] All targets already met! Nothing to do.")
        return

    # ── Connect to HF stream ──────────────────────────────────────────────
    print(f"\n[INFO] Logging in to HuggingFace...")
    from huggingface_hub import login
    login(token=args.hf_token)

    print(f"[INFO] Connecting to HF stream... Hunting for ≥{MIN_DURATION_SECONDS}s videos.")
    print(f"[INFO] Targets: {TARGETS}")
    dataset = load_dataset(args.hf_dataset, split="train", streaming=True)

    # ── Stream & download loop ─────────────────────────────────────────────
    stream_valid_hits = {cat: 0 for cat in target_categories}
    skipped_short = {cat: 0 for cat in target_categories}
    skipped_other = 0

    for i, sample in enumerate(dataset):
        metadata = sample.get('json', {})
        category = metadata.get('content_parent_category')
        duration = metadata.get('duration_seconds', 0)

        if category not in target_categories:
            skipped_other += 1
            if skipped_other % 2000 == 0:
                print(f"  [Scan] Skipped {skipped_other} non-target entries...")
            continue

        # Check duration threshold
        if duration < MIN_DURATION_SECONDS:
            skipped_short[category] += 1
            if skipped_short[category] % 100 == 0:
                print(f"  [Hunt] Skipped {skipped_short[category]} short videos in {category}...")
            continue

        # Valid video found! Check if we already have it (fast-forward)
        if stream_valid_hits[category] < existing_counts[category]:
            stream_valid_hits[category] += 1
            continue

        # Check if we still need more of this category
        if stream_valid_hits[category] >= TARGETS[category]:
            # Check if ALL categories are done
            if all(stream_valid_hits[cat] >= TARGETS[cat] for cat in target_categories):
                print("\n[INFO] All targets reached!")
                break
            continue

        # ── Download this video ────────────────────────────────────────────
        safe_cat_name = "edu" if category == "Education" else "sci"
        video_filename = f"fv_long_{safe_cat_name}_{stream_valid_hits[category]}.mp4"
        absolute_video_path = os.path.join(vids_dir, video_filename)

        print(f"[{safe_cat_name} {stream_valid_hits[category]+1}/{TARGETS[category]}] "
              f"Downloading {category} ({duration/60:.1f} min): {video_filename}...")

        # Write mp4 to disk
        with open(absolute_video_path, 'wb') as f:
            f.write(sample['mp4'])

        # ── Build QA conversations (same format as original) ───────────────
        conversations = []
        content_meta = metadata.get("content_metadata", {})

        # Part A: Storyline Summary
        storyline = content_meta.get("storylines", {}).get("description", "")
        if storyline:
            conversations.append({"from": "human", "value": "Can you provide a detailed summary of the storyline in this video?\n<video>"})
            conversations.append({"from": "gpt", "value": storyline})

        # Part B: Characters
        chars = content_meta.get("characterList", [])
        if chars:
            char_text = ", ".join([c.get("description", c.get("name", ""))
                                   for c in chars if c.get("description")])
            if char_text:
                prompt_text = "Who are the main characters in this video and what do they look like?"
                if not conversations:
                    prompt_text += "\n<video>"
                conversations.append({"from": "human", "value": prompt_text})
                conversations.append({"from": "gpt", "value": f"The video features the following characters: {char_text}."})

        # Part C: The Explicit Q&A Pairs
        qa_list = content_meta.get("qAndA", [])
        for qa in qa_list:
            question = qa.get("question")
            answer = qa.get("answer")
            if question and answer:
                if not conversations:
                    question += "\n<video>"
                conversations.append({"from": "human", "value": question})
                conversations.append({"from": "gpt", "value": answer})

        # ── Append + aggressive save ───────────────────────────────────────
        # Store youtube_id + duration so we can always trace back to HF entry
        original_filename = metadata.get("original_video_filename", "")
        youtube_id = original_filename.replace(".mp4", "") if original_filename else ""

        if conversations:
            sft_data.append({
                "id": f"fv_long_{safe_cat_name}_{stream_valid_hits[category]}",
                "video": absolute_video_path,
                "conversations": conversations,
                "youtube_id": youtube_id,
                "original_filename": original_filename,
                "duration_seconds": duration,
                "content_category": category,
                "content_fine_category": metadata.get("content_fine_category", ""),
            })

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(sft_data, f, indent=2, ensure_ascii=False)

        stream_valid_hits[category] += 1

        # Status every 10 videos
        if (stream_valid_hits["Education"] + stream_valid_hits["Science & Technology"]) % 10 == 0:
            print(f"  [Status] edu={stream_valid_hits['Education']}/{TARGETS['Education']} "
                  f"sci={stream_valid_hits['Science & Technology']}/{TARGETS['Science & Technology']} "
                  f"json={len(sft_data)}")

    # ── Summary ────────────────────────────────────────────────────────────
    # Count QA stats
    qa_counts = [len(e["conversations"]) // 2 for e in sft_data]
    total_qa = sum(qa_counts)

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos in JSON: {len(sft_data)}")
    print(f"Total QA pairs: {total_qa}")
    print(f"Avg QA per video: {total_qa/len(sft_data):.1f}" if sft_data else "")
    print(f"Videos saved to: {vids_dir}")
    print(f"JSON saved to: {output_json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
