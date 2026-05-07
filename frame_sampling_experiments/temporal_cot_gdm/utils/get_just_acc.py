import json
from pathlib import Path

# === SET YOUR PATH HERE ===
# You can put a path to a single .jsonl file OR a folder containing them.
TARGET_PATH = rf"/workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/results/lvbench_v2" 
# ==========================

def calculate_accuracy(file_path):
    """Reads a JSONL file and returns the correct predictions and total valid lines."""
    correct = 0
    total = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                predicted = data.get("predicted_letter")
                ground_truth = data.get("ground_truth")
                
                # Only count lines that actually have both keys
                if predicted is not None and ground_truth is not None:
                    total += 1
                    # Ensure case-insensitive comparison just in case
                    if str(predicted).strip().upper() == str(ground_truth).strip().upper():
                        correct += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {file_path}")
                
    return correct, total

def main():
    input_path = Path(TARGET_PATH)
    jsonl_files = []

    # Determine if input is a file or directory
    if input_path.is_file() and input_path.suffix == '.jsonl' and input_path.prefix == "Qwen3":
        jsonl_files.append(input_path)
    elif input_path.is_dir():
        # Recursively find all .jsonl files in the directory
        jsonl_files = list(input_path.rglob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in directory: {input_path}")
            return
    else:
        print(f"Error: Invalid path or not a .jsonl file -> {input_path}")
        return

    total_correct = 0
    total_samples = 0

    print(f"\nProcessing {len(jsonl_files)} file(s) from: {input_path}\n")
    print("-" * 50)

    # Process each file and calculate individual accuracies
    for file_path in jsonl_files:
        if "_v7.jsonl" in file_path.name:
            correct, total = calculate_accuracy(file_path)
            total_correct += correct
            total_samples += total
            
            if total > 0:
                accuracy = (correct / total) * 100
                print(f"{file_path.name}: {accuracy:.2f}% ({correct}/{total})")
            else:
                print(f"{file_path.name}: No valid evaluation data found.")
    
    print("-" * 50)

    # Print overall accuracy if processing multiple files
    if len(jsonl_files) > 1:
        if total_samples > 0:
            overall_accuracy = (total_correct / total_samples) * 100
            print(f"OVERALL ACCURACY: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
        else:
            print("OVERALL ACCURACY: No valid evaluation data found across all files.")
    print("\n")

if __name__ == "__main__":
    main()