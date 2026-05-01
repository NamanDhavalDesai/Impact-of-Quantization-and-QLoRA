"""
Post-Training Analysis Script
Generates publication-ready figures and tables from experiment results.

Usage:
    python scripts/generate_figures.py

Output:
    - figures/performance_comparison.png
    - figures/degradation_analysis.png
    - figures/confusion_matrices.png
    - figures/swedish_recovery.png
    - tables/full_results.csv
    - tables/summary_statistics.csv
"""

import os
import json
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def load_all_results(base_dir="results"):
    """Load all result files and associated configs."""
    files = glob.glob(f"{base_dir}/**/results_*.json", recursive=True)
    
    all_results = []
    
    for fpath in files:
        try:
            with open(fpath, "r") as f:
                res = json.load(f)
            
            # Get config
            dir_path = os.path.dirname(fpath)
            config_path = os.path.join(dir_path, ".hydra", "config.yaml")
            
            if os.path.exists(config_path):
                with open(config_path, "r") as cf:
                    cfg = yaml.safe_load(cf)
                
                res['_model'] = cfg['model']['name']
                res['_task'] = cfg['task']['name']
                res['_config_path'] = config_path
            
            res['_file_path'] = fpath
            all_results.append(res)
            
        except Exception as e:
            print(f"Warning: Could not load {fpath}: {e}")
    
    print(f"Loaded {len(all_results)} result files")
    return all_results


def create_performance_table(results):
    """Create a DataFrame with all performance metrics."""
    data = defaultdict(dict)
    
    for res in results:
        model = res.get('_model', res.get('metadata', {}).get('model_name', 'unknown'))
        task = res.get('_task', res.get('metadata', {}).get('task_name', 'unknown'))
        
        data[model][task] = {
            'macro_f1': res.get('macro_f1', 0),
            'accuracy': res.get('accuracy', 0),
            'num_samples': res.get('num_samples', len(res.get('predictions', []))),
        }
    
    # Create multi-level DataFrame
    rows = []
    for model, tasks in data.items():
        for task, metrics in tasks.items():
            rows.append({
                'Model': model,
                'Task': task,
                'Macro F1': metrics['macro_f1'],
                'Accuracy': metrics['accuracy'],
                'Samples': metrics['num_samples'],
            })
    
    df = pd.DataFrame(rows)
    return df


def plot_performance_comparison(df, output_dir="figures"):
    """Bar chart comparing models across all tasks."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Pivot for plotting
    pivot = df.pivot(index='Task', columns='Model', values='Macro F1')
    
    # Reorder columns
    model_order = ['16bit', '8bit', '4bit', '4bit_adapter']
    existing = [m for m in model_order if m in pivot.columns]
    pivot = pivot[existing]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Macro F1 Score')
    ax.set_xlabel('Task')
    ax.set_title('Model Performance Comparison Across Tasks')
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8, rotation=90, padding=3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/performance_comparison.png")


def plot_degradation_analysis(df, output_dir="figures"):
    """Show performance degradation from 16bit to 4bit."""
    os.makedirs(output_dir, exist_ok=True)
    
    pivot = df.pivot(index='Task', columns='Model', values='Macro F1')
    
    if '16bit' not in pivot.columns or '4bit' not in pivot.columns:
        print("Warning: Need both 16bit and 4bit results for degradation analysis")
        return
    
    # Calculate degradation
    degradation = ((pivot['16bit'] - pivot['4bit']) / pivot['16bit'] * 100).fillna(0)
    
    # Separate English and Swedish
    colors = ['#2ecc71' if 'en' in task else '#e74c3c' for task in degradation.index]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(degradation.index, degradation.values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Performance Drop (%)')
    ax.set_xlabel('Task')
    ax.set_title('Quantization Degradation: 16-bit → 4-bit')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='English'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Swedish'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add value labels
    for bar, val in zip(bars, degradation.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/degradation_analysis.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/degradation_analysis.png")


def plot_swedish_recovery(df, output_dir="figures"):
    """Show how M4 (QLoRA) recovers Swedish performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to Swedish tasks
    sv_df = df[df['Task'].str.contains('sv')]
    
    if sv_df.empty:
        print("Warning: No Swedish results found")
        return
    
    pivot = sv_df.pivot(index='Task', columns='Model', values='Macro F1')
    
    # Reorder
    model_order = ['16bit', '8bit', '4bit', '4bit_adapter']
    existing = [m for m in model_order if m in pivot.columns]
    pivot = pivot[existing]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(pivot.index))
    width = 0.2
    
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#2ecc71']
    
    for i, (model, color) in enumerate(zip(existing, colors)):
        bars = ax.bar(x + i*width, pivot[model], width, label=model, color=color, edgecolor='black', linewidth=0.5)
        # Add value labels
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Macro F1 Score')
    ax.set_xlabel('Task')
    ax.set_title('Swedish Task Performance: Quantization Impact & QLoRA Recovery')
    ax.set_xticks(x + width * (len(existing)-1) / 2)
    ax.set_xticklabels(pivot.index)
    ax.legend(title='Model')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/swedish_recovery.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/swedish_recovery.png")


def plot_confusion_matrices(results, output_dir="figures"):
    """Plot confusion matrices for all model/task combinations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by task
    tasks = set()
    for res in results:
        task = res.get('_task', res.get('metadata', {}).get('task_name'))
        if task:
            tasks.add(task)
    
    for task in sorted(tasks):
        task_results = [r for r in results if r.get('_task') == task or 
                        r.get('metadata', {}).get('task_name') == task]
        
        if not task_results:
            continue
        
        n_models = len(task_results)
        fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for ax, res in zip(axes, sorted(task_results, key=lambda x: x.get('_model', ''))):
            model = res.get('_model', res.get('metadata', {}).get('model_name', 'unknown'))
            cm = np.array(res.get('confusion_matrix', []))
            labels = res.get('labels_order', [])
            
            if cm.size == 0:
                continue
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=labels, yticklabels=labels)
            ax.set_title(f'{model}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        fig.suptitle(f'Confusion Matrices: {task}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_{task}.png", bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/confusion_{task}.png")


def save_tables(df, results, output_dir="tables"):
    """Save CSV tables for the report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Full results
    df.to_csv(f"{output_dir}/full_results.csv", index=False)
    print(f"Saved: {output_dir}/full_results.csv")
    
    # Pivot table (what you'll put in the paper)
    pivot = df.pivot(index='Model', columns='Task', values='Macro F1')
    model_order = ['16bit', '8bit', '4bit', '4bit_adapter']
    existing = [m for m in model_order if m in pivot.index]
    pivot = pivot.reindex(existing)
    pivot.to_csv(f"{output_dir}/performance_matrix.csv")
    print(f"Saved: {output_dir}/performance_matrix.csv")
    
    # Per-class F1 scores
    per_class_data = []
    for res in results:
        model = res.get('_model', res.get('metadata', {}).get('model_name', 'unknown'))
        task = res.get('_task', res.get('metadata', {}).get('task_name', 'unknown'))
        per_class = res.get('per_class_f1', {})
        for label, f1 in per_class.items():
            per_class_data.append({
                'Model': model,
                'Task': task, 
                'Label': label,
                'F1': f1
            })
    
    if per_class_data:
        per_class_df = pd.DataFrame(per_class_data)
        per_class_df.to_csv(f"{output_dir}/per_class_f1.csv", index=False)
        print(f"Saved: {output_dir}/per_class_f1.csv")


def main():
    print("=" * 60)
    print("Post-Training Analysis: Generating Figures and Tables")
    print("=" * 60)
    
    # Load results
    results = load_all_results()
    
    if not results:
        print("No results found! Run experiments first.")
        return
    
    # Create performance table
    df = create_performance_table(results)
    print("\n=== Performance Summary ===")
    print(df.to_string(index=False))
    
    # Generate figures
    print("\n=== Generating Figures ===")
    plot_performance_comparison(df)
    plot_degradation_analysis(df)
    plot_swedish_recovery(df)
    plot_confusion_matrices(results)
    
    # Save tables
    print("\n=== Saving Tables ===")
    save_tables(df, results)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("  Figures saved to: figures/")
    print("  Tables saved to: tables/")
    print("=" * 60)


if __name__ == "__main__":
    main()
