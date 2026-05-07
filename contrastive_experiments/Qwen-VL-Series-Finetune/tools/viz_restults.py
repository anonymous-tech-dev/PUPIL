import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = "/workspace/Pupil/contrastive_experiments/outputs"
RUNS = [
    "contrastive_sft_v01_run1_batch_f8_bs32",
    "contrastive_sft_v02_run1_blackened_f8_bs32",
    "contrastive_sft_v03_run1_gaussian_f8_bs32",
    "contrastive_sft_v01_run3_batch_f16_bs32",
    "contrastive_sft_v02_run3_blackened_f16_bs32",
    "contrastive_sft_v03_run3_gaussian_f16_bs32",
]

# Color map for strategies, Line style for frames
STYLE_MAP = {
    "batch": {"color": "#1f77b4"},      # Blue
    "blackened": {"color": "#d62728"},  # Red
    "gaussian": {"color": "#2ca02c"},   # Green
}

def smooth_curve(scalars, weight=0.85):
    """Exponential Moving Average smoothing (matches TensorBoard)."""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def extract_tb_data(run_path, tag):
    """Finds the tfevents file and extracts the specified scalar tag."""
    search_pattern = os.path.join(run_path, "**", "events.out.tfevents.*")
    event_files = glob.glob(search_pattern, recursive=True)
    
    if not event_files:
        print(f"⚠️ No TensorBoard events found in {run_path}")
        return [], []
    
    available_tags = set()
    
    # Check all event files (newest first)
    for event_file in sorted(event_files, reverse=True):
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            # Check what scalar tags are in this specific file
            scalars = ea.Tags().get('scalars', [])
            available_tags.update(scalars)
            
            if tag in scalars:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                return steps, vals
        except Exception as e:
            continue
            
    print(f"⚠️ Tag '{tag}' NOT FOUND in {run_path}. Available tags are: {available_tags}")
    return [], []

# ==========================================
# PLOTTING
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid') # Clean, modern grid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for run_name in RUNS:
    run_path = os.path.join(BASE_DIR, run_name)
    
    # Parse the run name for styling
    strategy = "batch" if "batch" in run_name else "blackened" if "blackened" in run_name else "gaussian"
    frames = "f8" if "f8" in run_name else "f16"
    
    color = STYLE_MAP[strategy]["color"]
    linestyle = "-" if frames == "f8" else "--"
    label = f"{strategy.capitalize()} ({frames})"
    
    # 1. Plot Total Loss
    steps, total_loss = extract_tb_data(run_path, "train/loss/total") 
    if total_loss:
        smoothed_total = smooth_curve(total_loss)
        ax1.plot(steps, smoothed_total, color=color, linestyle=linestyle, linewidth=2, label=label)
        # Plot raw loss very faintly in the background
        ax1.plot(steps, total_loss, color=color, alpha=0.15, linestyle=linestyle)

    # 2. Plot Contrastive Loss
    # 2. Plot Contrastive Loss
    steps, contrastive_loss = extract_tb_data(run_path, "train/loss/contrastive") 
    if contrastive_loss:
        smoothed_contrastive = smooth_curve(contrastive_loss)
        ax2.plot(steps, smoothed_contrastive, color=color, linestyle=linestyle, linewidth=2, label=label)
        ax2.plot(steps, contrastive_loss, color=color, alpha=0.15, linestyle=linestyle)

# Formatting Ax1 (Total Loss)
ax1.set_title("Total Training Loss (SFT + λ*Contrastive)", fontsize=14, pad=15)
ax1.set_xlabel("Training Steps", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.legend(loc="upper right", frameon=True)

# Formatting Ax2 (Contrastive Loss)
ax2.set_title("Contrastive Loss Only", fontsize=14, pad=15)
ax2.set_xlabel("Training Steps", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.legend(loc="upper right", frameon=True)

plt.tight_layout()
output_path = os.path.join(BASE_DIR, "loss_curves_summary.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Success! Beautiful chart saved to: {output_path}")