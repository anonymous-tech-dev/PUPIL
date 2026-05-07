"""
Compare results from multiple contrastive learning experiments.
Generates comparison table and plots.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_experiment_metrics(output_dir: str) -> Dict[str, Dict]:
    """Load metrics from all experiments in output directory."""
    experiments = {}
    
    output_path = Path(output_dir)
    for exp_dir in output_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        metrics_file = exp_dir / "test_results" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            experiments[exp_dir.name] = metrics
    
    return experiments


def create_comparison_table(experiments: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table from experiment metrics."""
    data = []
    
    for exp_name, metrics in experiments.items():
        row = {"experiment": exp_name}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by experiment name
    df = df.sort_values("experiment")
    
    # Reorder columns
    metric_cols = [col for col in df.columns if col != "experiment"]
    df = df[["experiment"] + sorted(metric_cols)]
    
    return df


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON")
    print("="*100)
    
    # Print header
    header = "| " + " | ".join(f"{col:20s}" for col in df.columns) + " |"
    print(header)
    print("|" + "-"*98 + "|")
    
    # Print rows
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.4f}".ljust(20))
            else:
                values.append(str(val).ljust(20))
        print("| " + " | ".join(values) + " |")
    
    print("="*100 + "\n")


def save_comparison_csv(df: pd.DataFrame, output_path: str):
    """Save comparison table as CSV."""
    df.to_csv(output_path, index=False)
    print(f"Comparison table saved to: {output_path}")


def plot_metrics_comparison(df: pd.DataFrame, output_dir: str):
    """Plot metrics comparison across experiments."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Skipping plots.")
        return
    
    sns.set_style("whitegrid")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_cols:
        print("No numeric metrics to plot")
        return
    
    # Create subplots for each metric
    n_metrics = len(numeric_cols)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for idx, metric in enumerate(numeric_cols):
        ax = axes[idx] if n_metrics > 1 else axes[0]
        
        # Bar plot
        df.plot(
            x='experiment',
            y=metric,
            kind='bar',
            ax=ax,
            legend=False,
            color='steelblue'
        )
        
        ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Experiment', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    
    plt.close()


def find_best_experiment(df: pd.DataFrame, metric: str = "exact_match") -> str:
    """Find best performing experiment for a given metric."""
    if metric not in df.columns:
        print(f"Warning: Metric '{metric}' not found in results")
        return None
    
    best_idx = df[metric].idxmax()
    best_exp = df.loc[best_idx, "experiment"]
    best_score = df.loc[best_idx, metric]
    
    return best_exp, best_score


def main():
    parser = argparse.ArgumentParser(description="Compare contrastive learning experiments")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing experiment outputs"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="exact_match",
        help="Metric to use for finding best experiment"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save comparison table as CSV"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate comparison plots"
    )
    
    args = parser.parse_args()
    
    # Load experiment metrics
    print(f"Loading experiments from: {args.output_dir}")
    experiments = load_experiment_metrics(args.output_dir)
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"Found {len(experiments)} experiments")
    
    # Create comparison table
    df = create_comparison_table(experiments)
    
    # Print comparison
    print_comparison_table(df)
    
    # Find best experiment
    best_exp, best_score = find_best_experiment(df, args.metric)
    print(f"\nBest experiment (by {args.metric}): {best_exp} ({best_score:.4f})")
    
    # Save CSV
    if args.save_csv:
        csv_path = os.path.join(args.output_dir, "comparison.csv")
        save_comparison_csv(df, csv_path)
    
    # Generate plots
    if args.plot:
        plot_metrics_comparison(df, args.output_dir)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()