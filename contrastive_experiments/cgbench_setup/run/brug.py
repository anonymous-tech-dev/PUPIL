import json
import os

def generate_unified_json():
    # Your exact paths
    cgbench_path = "/workspace/Pupil/contrastive_experiments/cgbench_setup/cgbench.json"
    metadata_path = "/workspace/Pupil/contrastive_experiments/cgbench_setup/run/video_meta_info.json"
    output_path = "/workspace/Pupil/contrastive_experiments/cgbench_setup/unified_training_data.json"
    videos_dir = "/data/Pupil/CGBench"

    print("Loading original JSON files...")
    
    try:
        with open(cgbench_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {cgbench_path}")
        return

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            video_meta = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {metadata_path}")
        return

    merged_data = []
    missing_full_videos = 0
    missing_clue_videos = 0

    print("Merging data and mapping absolute video paths...")
    
    for item in qa_data:
        vid_id = item.get("video_uid")
        qid = item.get("qid")
        
        # Map both potential video files since they are in the same folder
        full_video_path = os.path.join(videos_dir, f"{vid_id}.mp4")
        clue_video_path = os.path.join(videos_dir, f"{qid}.mp4")
        
        full_exists = os.path.exists(full_video_path)
        clue_exists = os.path.exists(clue_video_path)
        
        if not full_exists: missing_full_videos += 1
        if not clue_exists: missing_clue_videos += 1

        meta = video_meta.get(vid_id, {})

        # Build the structured entry
        unified_entry = {
            "id": qid,
            "video_uid": vid_id,
            "paths": {
                "full_video": full_video_path,
                "full_video_exists": full_exists,
                "clue_video": clue_video_path,
                "clue_video_exists": clue_exists
            },
            "question": item.get("question"),
            "options": item.get("choices"),
            "answer_text": item.get("answer"),
            "answer_idx": item.get("right_answer"),
            "clue_intervals": item.get("clue_intervals", []),
            "metadata": {
                "duration": item.get("duration", meta.get("duration")),
                "fps": meta.get("fps"),
                "max_frame": meta.get("max_frame"),
                "domain": item.get("domain"),
                "sub_category": item.get("sub_category")
            }
        }

        merged_data.append(unified_entry)

    # Save it out
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4)

    print(f"\nDone! Unified JSON successfully saved to:\n{output_path}\n")
    print(f"Total QA pairs processed: {len(merged_data)}")
    print(f"Full videos missing from directory: {missing_full_videos}")
    print(f"Clue videos missing from directory: {missing_clue_videos}")

if __name__ == "__main__":
    generate_unified_json()