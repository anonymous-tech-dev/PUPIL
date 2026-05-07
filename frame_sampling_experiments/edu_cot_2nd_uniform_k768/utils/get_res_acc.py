import json
import re
from pathlib import Path

# === SET YOUR PATH HERE ===
TARGET_PATH = r"/workspace/Pupil/frame_sampling_experiments/edu_cot/results/lvbench_v2"
# ==========================


def extract_answer_letter(raw: str) -> str:
    """Extract a single answer letter (A-E) from raw VLM output."""
    raw = raw.strip()
    if not raw:
        return ""
    # Bare letter
    if raw[0] in "ABCDE":
        return raw[0]
    # Parenthesised letter
    m = re.search(r"\(([A-E])\)", raw)
    if m:
        return m.group(1)
    # "answer is X" / "option X"
    m = re.search(
        r"(?:answer|option)\s*(?:is\s*)?[:\s]*\(?([A-E])\)?",
        raw, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    # Last resort: first letter found
    for ch in raw:
        if ch in "ABCDE":
            return ch
    return ""


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
                predicted = data.get("predicted_letter", "")
                ground_truth = data.get("ground_truth", "")

                # If predicted_letter is empty, try extracting from raw_answer
                if not predicted:
                    raw = data.get("raw_answer", "")
                    predicted = extract_answer_letter(raw)

                if predicted and ground_truth:
                    total += 1
                    if str(predicted).strip().upper() == str(ground_truth).strip().upper():
                        correct += 1
                else:
                    total += 1  # count it but as wrong

            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {file_path}")

    return correct, total


def main():
    input_path = Path(TARGET_PATH)
    jsonl_files = []

    if input_path.is_file() and input_path.suffix == '.jsonl':
        jsonl_files.append(input_path)
    elif input_path.is_dir():
        jsonl_files = sorted(input_path.rglob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in directory: {input_path}")
            return
    else:
        print(f"Error: Invalid path or not a .jsonl file -> {input_path}")
        return

    total_correct = 0
    total_samples = 0

    print(f"\nProcessing {len(jsonl_files)} file(s) from: {input_path}\n")
    print("-" * 60)

    for file_path in sorted(jsonl_files):
        correct, total = calculate_accuracy(file_path)
        total_correct += correct
        total_samples += total

        if total > 0:
            accuracy = (correct / total) * 100
            print(f"{file_path.name}: {accuracy:.2f}% ({correct}/{total})")
        else:
            print(f"{file_path.name}: No valid evaluation data found.")

    print("-" * 60)

    if len(jsonl_files) > 1:
        if total_samples > 0:
            overall_accuracy = (total_correct / total_samples) * 100
            print(f"OVERALL ACCURACY: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
        else:
            print("OVERALL ACCURACY: No valid evaluation data found across all files.")
    print("\n")


if __name__ == "__main__":
    main()
