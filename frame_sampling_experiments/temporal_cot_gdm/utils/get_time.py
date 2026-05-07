import json
import os

# Set your file path through this variable
file_path = "/home/Pupil/frame_sampling_experiments/temporal_cot_gdm/results/lvbench/Qwen2.5-VL-7B_dynamic_segment_results_old.jsonl" 

def get_time_stats(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Could not find file at '{filepath}'")
        return

    total_time = 0.0
    count = 0

    try:
        # First, try loading it as a standard JSON array [ {...}, {...} ]
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            if isinstance(data, list):
                for entry in data:
                    if 'time_taken_secs' in entry:
                        total_time += entry['time_taken_secs']
                        count += 1
            elif isinstance(data, dict) and 'time_taken_secs' in data:
                # In case the file is just one single JSON object
                total_time += data['time_taken_secs']
                count += 1

    except json.JSONDecodeError:
        # If standard JSON fails, try reading it as JSONL (one JSON object per line)
        total_time = 0.0
        count = 0
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if 'time_taken_secs' in entry:
                        total_time += entry['time_taken_secs']
                        count += 1
                except json.JSONDecodeError:
                    continue # Skip invalid lines

    # Calculate and print results
    if count == 0:
        print("Could not find any entries with 'time_taken_secs' in the file.")
        return

    avg_time = total_time / count

    print("-" * 30)
    print(f"Total entries processed: {count}")
    print(f"Total time taken:        {total_time:.2f} seconds")
    print(f"Average time taken:      {avg_time:.2f} seconds")
    print("-" * 30)

# Run the function
get_time_stats(file_path)