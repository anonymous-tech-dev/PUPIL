import os
import json
import pandas as pd

def compile_evaluation_results(base_dir="outputs"):
    results = []

    # Iterate through all folders in the outputs directory
    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(folder_path):
            continue

        # Determine the correct path to the JSON file based on the folder name
        if folder_name.startswith("baseline_"):
            json_path = os.path.join(folder_path, "evaluation_results.json")
        elif folder_name.startswith("contrastive_sft_"):
            json_path = os.path.join(folder_path, "test_results", "evaluation_results.json")
        else:
            continue # Skip any unrelated folders

        # Check if the file exists before trying to read it
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                metrics = data.get("metrics", {})
                
                # Extract metrics and format the row
                row = {
                    "Run/Model Name": folder_name,
                    "Exact Match": metrics.get("exact_match"),
                    "Token Acc": metrics.get("token_accuracy"),
                    "BLEU": metrics.get("bleu"),
                    "ROUGE-1": metrics.get("rouge1"),
                    "ROUGE-2": metrics.get("rouge2"),
                    "ROUGE-L": metrics.get("rougeL"),
                    "LLM Judge": metrics.get("llm_judge_score")
                }
                results.append(row)
                
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON in {json_path}")
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
        else:
            print(f"Warning: File not found -> {json_path}")

    # Create a pandas DataFrame
    df = pd.DataFrame(results)
    
    # Optional: Round the numerical columns for a cleaner table display
    numeric_cols = ["Exact Match", "Token Acc", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "LLM Judge"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(4)

    return df

if __name__ == "__main__":
    # Ensure this script is run from the directory containing the 'outputs' folder
    # or change 'outputs' to your absolute path (e.g., '/path/to/your/outputs')
    df_results = compile_evaluation_results("/workspace/Pupil/contrastive_experiments/outputs")
    
    # Print as a formatted string in the console
    print("\n--- Evaluation Metrics Summary ---")
    print(df_results.to_string(index=False))
    
    # Save the table to a CSV file for easy viewing in Excel/Google Sheets
    output_csv = "metrics_summary_table.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved summary table to {output_csv}")