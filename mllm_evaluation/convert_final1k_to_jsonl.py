#!/usr/bin/env python3
"""
Convert final_consolidated_1k.json to the JSONL format expected by
EduCoT and TCoT pipelines.

Output schema (one JSON object per line):
{
  "uid": "query_id",
  "video": "filename.mp4",
  "question": "...",
  "answer_choices": [],       # empty for open-ended
  "ground_truth": "...",      # free-form text
  "question_type": ["pipeline_mode"],
  "time_reference": "",
  "source_of_fact": "audio|visual|time|priority",
  "category": "cognitive_category"
}
"""
import json, sys, os

INPUT = os.environ.get(
    "INPUT_JSON",
    "/workspace/Pupil/dataset_curation/dataset/queries_db/final_1k/final_consolidated_1k.json",
)
OUTPUT = os.environ.get(
    "OUTPUT_JSONL",
    "/workspace/Pupil/dataset_curation/dataset/queries_db/final_1k/final_1k_for_cot.jsonl",
)

data = json.load(open(INPUT))
count = 0
with open(OUTPUT, "w") as f:
    for video_key in sorted(data.keys()):
        for q in data[video_key]:
            item = {
                "uid": q.get("query_id", f"q_{count}"),
                "video": video_key,
                "question": q.get("question", ""),
                "answer_choices": [],
                "ground_truth": q.get("ground_truth", ""),
                "question_type": [q.get("annotations", {}).get("pipeline_mode", "unknown")],
                "time_reference": "",
                "source_of_fact": q.get("annotations", {}).get("pipeline_mode", "unknown"),
                "category": q.get("annotations", {}).get("cognitive_category", "unknown"),
            }
            f.write(json.dumps(item) + "\n")
            count += 1

print(f"✅ Wrote {count} items to {OUTPUT}")
