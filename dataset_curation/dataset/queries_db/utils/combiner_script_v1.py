import json
import glob
import os

def yoink_and_merge(input_folder, output_file_path, mode='dict'):
    """
    Reads all JSON files from the input_folder and merges them.
    
    Args:
        mode (str): 
            'dict'        : (Standard) Merges top-level keys. WARNING: Overwrites value if key exists.
            'list'        : (Standard) Merges top-level arrays.
            'dict_append' : (Smart) Expects {key: [list]}. If key exists, appends to the list. 
                            Use this for combining "Video Path -> [Questions]" files.
    """
    
    # Initialize container
    if mode == 'list':
        merged_data = []
        print("--- Mode: LIST merging (concatenating arrays) ---")
    else:
        # Both 'dict' and 'dict_append' start with a dictionary
        merged_data = {} 
        if mode == 'dict_append':
            print("--- Mode: DICTIONARY APPEND (merging keys + extending list values) ---")
        else:
            print("--- Mode: DICTIONARY UPDATE (merging keys + overwriting values) ---")

    # Check if input directory exists
    if not os.path.isdir(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        return

    # Create a pattern to find all .json files in the folder
    search_pattern = os.path.join(input_folder, "*.json")
    json_files = glob.glob(search_pattern)

    print(f"Found {len(json_files)} JSON files. Starting the yoink process...")

    for file_path in json_files:
        # SAFETY CHECK: Don't read the output file if it sits in the same folder!
        if os.path.abspath(file_path) == os.path.abspath(output_file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # --- CASE 1: STANDARD DICT (Overwrite duplicates) ---
                if mode == 'dict':
                    if isinstance(data, dict):
                        merged_data.update(data)
                    else:
                        print(f"Skipping {os.path.basename(file_path)}: Expected 'dict' but found '{type(data).__name__}'.")

                # --- CASE 2: LIST MODE (Concat root arrays) ---
                elif mode == 'list':
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        print(f"Skipping {os.path.basename(file_path)}: Expected 'list' but found '{type(data).__name__}'.")
                
                # --- CASE 3: DICT APPEND (The "Combiner Combiner") ---
                elif mode == 'dict_append':
                    if isinstance(data, dict):
                        for key, value_list in data.items():
                            # If key exists, extend the list
                            if key in merged_data:
                                if isinstance(merged_data[key], list) and isinstance(value_list, list):
                                    merged_data[key].extend(value_list)
                                else:
                                    print(f"Warning: Type mismatch for key '{key}' in {os.path.basename(file_path)}. Overwriting.")
                                    merged_data[key] = value_list
                            # If key is new, just add it
                            else:
                                merged_data[key] = value_list
                    else:
                        print(f"Skipping {os.path.basename(file_path)}: Expected 'dict' but found '{type(data).__name__}'.")
                    
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred with {file_path}: {e}")

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the combined data to the new location
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=4)
        print(f"Success! Merged data ({mode} mode) saved to: {output_file_path}")
    except Exception as e:
        print(f"Failed to write output file: {e}")

# --- CONFIGURATION ---

INPUT_FOLDER = '/home/Pupil/dataset_curation/dataset/queries_db/v1_500/parity_100/' 
OUTPUT_LOCATION = '/home/Pupil/dataset_curation/dataset/queries_db/v1_500/parity_100/final_combined.json'

# MODES:
# 'list'        -> combiner v1
# 'dict'        -> combiner v2
# 'dict_append' -> combiner of combiner
MERGE_MODE = 'dict_append' 

if __name__ == "__main__":
    yoink_and_merge(INPUT_FOLDER, OUTPUT_LOCATION, MERGE_MODE)