import os
import json
import glob
from collections import Counter

def analyze_dataset_stats(folder_path):
    """
    Analyzes JSON files ending in '_queries.json' in the given folder 
    and prints statistics about questions, answers, and annotations.
    """
    
    # Initialize counters and accumulators
    total_queries = 0
    total_q_chars = 0
    total_a_chars = 0
    total_q_words = 0
    total_a_words = 0
    
    pipeline_mode_counts = Counter()
    cognitive_category_counts = Counter()
    
    # create the search pattern
    search_pattern = os.path.join(folder_path, "*_queries.json")
    files = glob.glob(search_pattern)
    
    print(f"Scanning folder: {folder_path}")
    print(f"Found {len(files)} files matching pattern '*_queries.json'.\n")
    
    if not files:
        print("No files found. Please check the directory path.")
        return

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # The structure is Dict[VideoPath, List[QueryObjects]]
            for video_path, queries_list in data.items():
                for query_obj in queries_list:
                    total_queries += 1
                    
                    # --- Extract Text ---
                    question_text = query_obj.get("question", "")
                    # Using 'ground_truth' as the answer field based on your example
                    answer_text = query_obj.get("ground_truth", "")
                    
                    # --- Calculate Lengths ---
                    total_q_chars += len(question_text)
                    total_a_chars += len(answer_text)
                    
                    # Word counts (splitting by whitespace)
                    total_q_words += len(question_text.split())
                    total_a_words += len(answer_text.split())
                    
                    # --- Extract Categories ---
                    # Safely get annotations dict, then keys
                    annotations = query_obj.get("annotations", {})
                    
                    p_mode = annotations.get("pipeline_mode", "N/A")
                    c_cat = annotations.get("cognitive_category", "N/A")
                    
                    pipeline_mode_counts[p_mode] += 1
                    cognitive_category_counts[c_cat] += 1
                    
        except json.JSONDecodeError:
            print(f"Skipping file due to JSON error: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # --- Calculation & Printing ---
    
    if total_queries == 0:
        print("No queries found in the dataset.")
        return

    avg_q_len_char = total_q_chars / total_queries
    avg_a_len_char = total_a_chars / total_queries
    avg_q_len_word = total_q_words / total_queries
    avg_a_len_word = total_a_words / total_queries

    print("=" * 50)
    print(" DATASET STATISTICS REPORT")
    print("=" * 50)
    print(f"Total Files Parsed:   {len(files)}")
    print(f"Total Queries Found:  {total_queries}")
    print("-" * 50)
    print("AVERAGE LENGTHS:")
    print(f"  Avg Question Length (chars): {avg_q_len_char:.2f}")
    print(f"  Avg Answer Length (chars):   {avg_a_len_char:.2f}")
    print(f"  Avg Question Words:          {avg_q_len_word:.2f}")
    print(f"  Avg Answer Words:            {avg_a_len_word:.2f}")
    print("-" * 50)
    print("PIPELINE MODE DISTRIBUTION:")
    for mode, count in pipeline_mode_counts.most_common():
        print(f"  {mode:<20}: {count}")
    print("-" * 50)
    print("COGNITIVE CATEGORY DISTRIBUTION:")
    for cat, count in cognitive_category_counts.most_common():
        print(f"  {cat:<20}: {count}")
    print("=" * 50)

# --- CONFIGURATION ---
# Replace this path with the actual path to your folder containing the .json files
folder_path = '/home/Pupil/dataset_curation/dataset/queries_db/v1_500/sof_priority'

if __name__ == "__main__":
    analyze_dataset_stats(folder_path)