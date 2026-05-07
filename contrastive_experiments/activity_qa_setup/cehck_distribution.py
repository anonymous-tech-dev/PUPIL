import os
import json

# --- PATHS (match your original script exactly) ---
DATA_ROOT = "/data/Pupil/ActivityNet-QA"
ANNO_DIR = os.path.join(DATA_ROOT, "annotations")
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")

# --- TARGET COUNTS ---
TARGETS = {
    "train": 1000,
    "val": 100,
    "test": 100
}

# --- 1. Collect downloaded video IDs ---
downloaded_videos = {
    os.path.splitext(f)[0]
    for f in os.listdir(VIDEO_DIR)
    if f.endswith(".mp4")
}

print(f"Downloaded videos detected: {len(downloaded_videos)}")

# --- 2. Process each split ---
for split, target in TARGETS.items():

    q_path = os.path.join(ANNO_DIR, f"{split}_q.json")
    a_path = os.path.join(ANNO_DIR, f"{split}_a.json")

    out_q_path = os.path.join(ANNO_DIR, f"{split}_q_1.json")
    out_a_path = os.path.join(ANNO_DIR, f"{split}_a_1.json")

    # Load original annotations
    with open(q_path, "r") as f:
        q_entries = json.load(f)

    with open(a_path, "r") as f:
        a_entries = json.load(f)

    # --- 3. Filter QA belonging to downloaded videos ---
    filtered_q = []
    filtered_a = []

    for q, a in zip(q_entries, a_entries):
        if q["video_name"] in downloaded_videos:
            filtered_q.append(q)
            filtered_a.append(a)

        if len(filtered_q) >= target:
            break

    # --- 4. Save WITHOUT touching originals ---
    with open(out_q_path, "w") as f:
        json.dump(filtered_q, f, indent=2)

    with open(out_a_path, "w") as f:
        json.dump(filtered_a, f, indent=2)

    # --- 5. Reporting ---
    unique_videos = {q["video_name"] for q in filtered_q}

    print(f"\n=== {split.upper()} SUBSET CREATED ===")
    print(f"QA pairs written:     {len(filtered_q)} / {target}")
    print(f"Unique videos used:   {len(unique_videos)}")
    if unique_videos:
        print(f"Avg QA per video:     {len(filtered_q)/len(unique_videos):.2f}")

print("\nDone. Original files untouched. New *_1.json files created.")
