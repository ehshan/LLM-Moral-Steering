import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Set global style
sns.set_theme(style="whitegrid")
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

def generate_analysis_report(df, output_dir, experiment_tag):
    """
    The Master Function.
    Takes raw results, calculates metrics, and routes to the correct visualisation suite.
    
    Args:
        df (pd.DataFrame): The results dataframe (Cols: Layer, Multiplier, Deon_Score, Util_Score, Invalid_Rate)
        output_dir (Path): Where to save images.
        experiment_tag (str): Label for the files (e.g., 'ai_judge_layer6').
    """
    print(f"Generating Analysis Report for: {experiment_tag}")
    
    # 1. Pre-Processing: Calculate the "Gap"
    # Gap > 0 means Deontological preference
    # Gap < 0 means Utilitarian preference
    df['Gap'] = df['Deon_Score'] - df['Util_Score']
    
    # 2. Determine Mode (Single Layer vs. Full Sweep)
    unique_layers = df['Layer'].nunique()
    is_full_sweep = unique_layers >= 30  # Allow for small margin of error (e.g. 0-31)
    
    print(f"   -> Detected {unique_layers} unique layers.")
    
    # 3. Standard Plots (Run for EVERY experiment)
    plot_s_curve(df, output_dir, experiment_tag)
    plot_health_line(df, output_dir, experiment_tag)
    
    # 4. Conditional Plots (Run only for Full Sweep)
    if is_full_sweep:
        print("   -> Mode: Full Sweep (Generating Heatmaps & Quartered Grids)")
        plot_global_heatmap(df, output_dir, experiment_tag)
        plot_refusal_wall(df, output_dir, experiment_tag)
        plot_quartered_analysis(df, output_dir, experiment_tag)
    else:
        print("   -> Mode: Precision Strike (Skipping Heatmaps & Quartered Grids)")
        # Optional: Add a specific "Single Layer Detail" plot here if needed
        
    print(f"Visualisation Complete. Saved to {output_dir}")


# --- INDIVIDUAL PLOTTING FUNCTIONS ---

def plot_s_curve(df, output_dir, tag):
    """Plots the standard S-Curve (Multiplier vs Deon Score)."""
    plt.figure(figsize=(10, 6))
    
    # If many layers, use hue. If one layer, just line.
    if df['Layer'].nunique() > 1:
        sns.lineplot(data=df, x='Multiplier', y='Deon_Score', hue='Layer', 
                     palette='viridis', marker='o', alpha=0.7, legend=False)
    else:
        sns.lineplot(data=df, x='Multiplier', y='Deon_Score', marker='o', linewidth=3, color='blue')
        
    # Add Reference Line (50% = Neutral)
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5, label="Neutral (50%)")
    
    plt.title(f'Steering Response S-Curve ({tag})', fontsize=14, weight='bold')
    plt.ylabel('Deontological Choice (%)')
    plt.xlabel('Steering Multiplier')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, f'Viz_SCurve_{tag}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_health_line(df, output_dir, tag):
    """Plots the Invalid Rate (Refusal Wall)."""
    plt.figure(figsize=(10, 6))
    
    if df['Layer'].nunique() > 1:
        sns.lineplot(data=df, x='Multiplier', y='Invalid_Rate', hue='Layer', 
                     palette='Reds', marker='o', alpha=0.7, legend=False)
    else:
        sns.lineplot(data=df, x='Multiplier', y='Invalid_Rate', marker='o', linewidth=3, color='red')

    # Add Danger Zone Line (10%)
    plt.axhline(10, color='black', linestyle='--', label="Unusable (>10%)")
    
    plt.title(f'Model Health / Refusal Wall ({tag})', fontsize=14, weight='bold')
    plt.ylabel('Invalid Rate (%)')
    plt.xlabel('Steering Multiplier')
    plt.ylim(0, 105)
    
    save_path = os.path.join(output_dir, f'Viz_Health_{tag}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_global_heatmap(df, output_dir, tag):
    """Plots Layer vs Multiplier for the 'Gap' metric."""
    plt.figure(figsize=(14, 10))
    
    pivot = df.pivot(index='Layer', columns='Multiplier', values='Gap')
    pivot.sort_index(inplace=True)
    
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="coolwarm", center=0,
                cbar_kws={'label': 'Deontological Advantage (%)'}, annot_kws={"size": 8})
    
    plt.title(f'Global Steering Heatmap ({tag})', fontsize=16, weight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'Viz_Global_Heatmap_{tag}.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_refusal_wall(df, output_dir, tag):
    """Plots Layer vs Multiplier for Invalid Rate."""
    plt.figure(figsize=(14, 10))
    
    pivot = df.pivot(index='Layer', columns='Multiplier', values='Invalid_Rate')
    pivot.sort_index(inplace=True)
    
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100,
                cbar_kws={'label': 'Invalid Rate (%)'}, annot_kws={"size": 8})
    
    plt.title(f'Refusal Wall Analysis ({tag})', fontsize=16, weight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'Viz_Refusal_Heatmap_{tag}.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_quartered_analysis(df, output_dir, tag):
    """Generates the 4x4 Grid breakdown (Only runs if we have enough layers)."""
    chunks = [
        (0, 7, "Quadrant 1: Early Layers (0-7)"),
        (8, 15, "Quadrant 2: Middle Layers (8-15)"),
        (16, 23, "Quadrant 3: Deep Layers (16-23)"),
        (24, 31, "Quadrant 4: Final Layers (24-31)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for i, (start, end, title) in enumerate(chunks):
        chunk_df = df[(df['Layer'] >= start) & (df['Layer'] <= end)]
        
        # S-Curve Subplot
        sns.lineplot(data=chunk_df, x='Multiplier', y='Deon_Score', hue='Layer',
                     palette='tab10', marker='o', ax=axes[i], legend='full')
        
        axes[i].set_title(title, fontsize=12, weight='bold')
        axes[i].set_ylim(-5, 105)
        axes[i].axhline(50, color='gray', linestyle='--')

    plt.suptitle(f'Quartered S-Curve Analysis ({tag})', fontsize=20, weight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'Viz_Quartered_{tag}.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()