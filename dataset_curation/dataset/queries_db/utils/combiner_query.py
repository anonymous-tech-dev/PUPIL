import os
import json
import glob
from collections import defaultdict

# Define your paths
BASE_DIR = "/home/Pupil/dataset_curation/dataset/queries_db/final_1k"
SUB_DIRS = ["sof_audio", "sof_priority", "sof_time", "sof_visual"]
OUTPUT_FILE = os.path.join(BASE_DIR, "final_consolidated_1k.json")

def process_datasets():
    # Dictionary to hold all queries mapped by video name
    consolidated_data = defaultdict(list)
    
    for sub_dir in SUB_DIRS:
        folder_path = os.path.join(BASE_DIR, sub_dir)
        
        # Find all query files in the current subdirectory
        query_files = glob.glob(os.path.join(folder_path, "*_queries.json"))
        
        for query_file in query_files:
            # Construct the expected metadata file name
            metadata_file = query_file.replace("_queries.json", "_metadata.jsonl")
            
            # Step 1: Read the metadata.jsonl and map query_ids to their cleaned segments
            metadata_map = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as mf:
                    for line in mf:
                        if not line.strip(): continue
                        meta_entry = json.loads(line)
                        q_id = meta_entry.get("linked_query_id")
                        
                        # Yoink the segments and drop the 'video_id' to avoid spam
                        raw_segments = meta_entry.get("retrieved_segments", [])
                        clean_segments = [
                            {"start": seg["start"], "end": seg["end"]} 
                            for seg in raw_segments 
                            if "start" in seg and "end" in seg
                        ]
                        metadata_map[q_id] = clean_segments
            else:
                print(f"Warning: Missing metadata file for {query_file}")
            
            # Step 2: Read the queries.json and attach the cleaned segments
            with open(query_file, 'r', encoding='utf-8') as qf:
                queries_data = json.load(qf)
                
                # The key in your json is the full path to the mp4
                for video_path, queries in queries_data.items():
                    # Extract just the video filename (e.g., "3_perplexing_physics_problems_clean.mp4")
                    video_name = os.path.basename(video_path)
                    
                    for q in queries:
                        q_id = q.get("query_id")
                        # Attach the segments from our metadata map (default to empty list if none found)
                        q["timestamp_segments"] = metadata_map.get(q_id, [])
                        
                        consolidated_data[video_name].append(q)
    
    # Step 3: Sort the dictionary alphabetically by video name
    sorted_consolidated_data = {
        vid_name: consolidated_data[vid_name]
        for vid_name in sorted(consolidated_data.keys())
    }
    
    # Step 4: Write it out to the final consolidated file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        json.dump(sorted_consolidated_data, out_f, indent=2)
        
    print(f"Done! Consolidated JSON saved to: {OUTPUT_FILE}")
    print(f"Total unique videos processed: {len(sorted_consolidated_data)}")

if __name__ == "__main__":
    process_datasets()