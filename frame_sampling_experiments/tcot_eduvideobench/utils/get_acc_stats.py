import json
import pandas as pd
import numpy as np

def load_results(filepath):
    """Loads JSON or JSONL evaluation files into a Pandas DataFrame."""
    data = []
    with open(filepath, 'r') as f:
        try:
            # Try to load as a single JSON array
            data = json.load(f)
        except json.JSONDecodeError:
            # Fallback to JSON Lines (JSONL)
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return pd.DataFrame(data)

def compare_experiments(file_paths, labels=None):
    if labels is None:
        labels = [f"Exp {i+1}" for i in range(len(file_paths))]
        
    all_metrics = []
    
    for path, label in zip(file_paths, labels):
        df = load_results(path)
        
        # 1. Calculate Accuracy
        # Ensure we strip any whitespace/parentheses from predicted letters if needed, 
        # though your ground_truths and predicted_letters look clean ("D", "C", etc.)
        df['is_correct'] = df['predicted_letter'] == df['ground_truth']
        accuracy = df['is_correct'].mean() * 100
        
        # 2. Clean placeholder values (-1 or -1.0 from baseline)
        df['pct_selected'] = df['pct_selected'].replace(-1.0, np.nan)
        df['total_frames'] = df['total_frames'].replace(-1, np.nan)
        
        # 3. Calculate Justification/TCoT metrics (if applicable)
        # Count how many justifications were generated on average per video
        if 'justifications' in df.columns:
            df['num_justifications'] = df['justifications'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            avg_justifications = df['num_justifications'].mean()
        else:
            avg_justifications = 0
            
        # 4. Aggregate metrics
        metrics = {
            "Experiment": label,
            "Stage Name": df['stage'].iloc[0] if 'stage' in df.columns else "unknown",
            "Total Samples": len(df),
            "Accuracy (%)": round(accuracy, 2),
            "Avg Time (s)": round(df['time_taken_secs'].mean(), 2),
            "Avg Context Frames": round(df['num_context'].mean(), 1),
            "Avg Total Frames": round(df['total_frames'].mean(), 1),
            "Avg Selected Frames": round(df['num_selected'].mean(), 1),
            "Avg % Selected": round(df['pct_selected'].mean(), 2),
            "Avg Justification Count": round(avg_justifications, 1)
        }
        
        all_metrics.append(metrics)
        
    # Compile into a single DataFrame for easy viewing
    comparison_df = pd.DataFrame(all_metrics)
    
    return comparison_df

if __name__ == "__main__":
    # --- UPDATE THESE PATHS TO YOUR ACTUAL FILE LOCATIONS ---
    file1 = "/home/Pupil/frame_sampling_experiments/temporal_cot_gdm/results/egoschema/Qwen2.5-VL-7B_baseline_native_results.jsonl"
    file2 = "/home/Pupil/frame_sampling_experiments/temporal_cot_gdm/results/egoschema/Qwen2.5-VL-7B_baseline_results.jsonl"
    file3 = "/home/Pupil/frame_sampling_experiments/temporal_cot_gdm/results/egoschema/Qwen2.5-VL-7B_dynamic_segment_results.jsonl"
    
    # You can comment out the file checking logic below if you are passing the right paths
    import os
    files_to_check = [file1, file2, file3]
    if all(os.path.exists(f) for f in files_to_check):
        results_df = compare_experiments(
            file_paths=[file1, file2, file3],
            labels=["Baseline Native", "TCoT Baseline", "TCoT Dynamic Segment"]
        )
        
        print("\n" + "="*80)
        print("EGOSCHEMA EXPERIMENT COMPARISON")
        print("="*80)
        # Transpose for easier reading if there are many columns, or print as a clean table
        print(results_df.to_string(index=False))
        print("="*80 + "\n")
    else:
        print("Please update the script with the correct paths to your JSON files to run the comparison.")