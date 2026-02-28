import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
        experiment_tag (str): Label for the files (e.g., 'v1_full_sweep').
    """
    print(f"Generating Analysis Report for: {experiment_tag}")
    
    # Ensure output_dir is a Path object for cross-platform compatibility
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-Processing: Calculate the "Gap" with MASKING
    # If Invalid_Rate is > 15%, we set the Gap to NaN. 
    # This forces Seaborn to draw a blank/gray square instead of a false "Neutral 0".
    df['Gap'] = np.where(
        df['Invalid_Rate'] > 15, 
        np.nan, 
        df['Deon_Score'] - df['Util_Score']
    )
    
    # Determine Mode (Single Layer vs. Full Sweep)
    unique_layers = sorted(df['Layer'].unique())
    is_full_sweep = len(unique_layers) >= 20  # Adjusted threshold
    
    print(f"   -> Detected {len(unique_layers)} unique layers.")
    
    # Dynamic Subsetting for Line Plots
    # If we have too many layers, the S-Curve becomes a spaghetti mess. 
    # We dynamically pick 4 evenly spaced layers to represent the trend.
    if len(unique_layers) > 6:
        idx = np.round(np.linspace(0, len(unique_layers) - 1, 4)).astype(int)
        target_layers = [unique_layers[i] for i in idx]
        print(f"   -> Subsetting line plots to representative layers: {target_layers}")
        line_df = df[df['Layer'].isin(target_layers)]
    else:
        line_df = df

    # Standard Plots (Run for EVERY experiment)
    plot_s_curve(line_df, output_dir, experiment_tag)
    plot_health_line(line_df, output_dir, experiment_tag)
    
    # Conditional Plots (Run only for massive sweeps)
    if is_full_sweep:
        print("   -> Mode: Full Sweep (Generating Heatmaps & Quartered Grids)")
        plot_global_heatmap(df, output_dir, experiment_tag)
        plot_refusal_wall(df, output_dir, experiment_tag)
        
        # Pass the full DF to the quartered functions (they handle their own chunking)
        plot_quartered_heatmap(df, 'Gap', 'Steering Effectiveness', 'Heatmap_Gap', 
                               cmap='coolwarm', center=0, fmt=".1f", output_dir=output_dir, tag=experiment_tag)
        plot_quartered_heatmap(df, 'Invalid_Rate', 'Refusal Wall', 'Heatmap_Invalid', 
                               cmap='Reds', vmin=0, vmax=100, fmt=".0f", output_dir=output_dir, tag=experiment_tag)
        plot_quartered_line(df, 'Deon_Score', 'S-Curve Response', 'SCurve', 
                            hline_val=50, output_dir=output_dir, tag=experiment_tag)
        plot_quartered_line(df, 'Invalid_Rate', 'Model Health', 'Health', 
                            hline_val=10, hline_color='red', output_dir=output_dir, tag=experiment_tag)
    else:
        print("   -> Mode: Precision Strike (Skipping Heatmaps & Quartered Grids)")
        
    print(f"Visualisation Complete. Saved to: {output_dir}")


# --- INDIVIDUAL PLOTTING FUNCTIONS ---

def plot_s_curve(df, output_dir, tag):
    """Plots the standard S-Curve (Multiplier vs Deon Score)."""
    plt.figure(figsize=(10, 6))

    # If many layers, use hue. If one layer, just line.
    if df['Layer'].nunique() > 1:
        sns.lineplot(data=df, x='Multiplier', y='Deon_Score', hue='Layer', 
                     palette='viridis', style='Layer', markers=True, dashes=False, linewidth=2.5)
    else:
        sns.lineplot(data=df, x='Multiplier', y='Deon_Score', marker='o', linewidth=3, color='blue')
        
    # Add Reference Line (50% = Neutral)
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5, label="Neutral (50%)")

    plt.title(f'Steering Response S-Curve ({tag})', fontsize=14, weight='bold')
    plt.ylabel('Deontological Choice (%)')
    plt.xlabel('Steering Multiplier')
    plt.tight_layout()

    plt.savefig(output_dir / f'Viz_SCurve_{tag}.png', dpi=300)
    plt.close()


def plot_health_line(df, output_dir, tag):
    """Plots the Invalid Rate (Refusal Wall)."""
    plt.figure(figsize=(10, 6))

    if df['Layer'].nunique() > 1:
        sns.lineplot(data=df, x='Multiplier', y='Invalid_Rate', hue='Layer', 
                     palette='Reds_r', style='Layer', markers=True, dashes=False, linewidth=2.5)
    else:
        sns.lineplot(data=df, x='Multiplier', y='Invalid_Rate', marker='o', linewidth=3, color='red')

    # Add Danger Zone Line (10%)
    plt.axhline(10, color='red', linestyle='--', alpha=0.5, label="Unusable (>10%)")
    
    plt.title(f'Model Health / Refusal Wall ({tag})', fontsize=14, weight='bold')
    plt.ylabel('Invalid Rate (%)')
    plt.xlabel('Steering Multiplier')
    plt.ylim(-5, 105)
    plt.tight_layout()
    
    plt.savefig(output_dir / f'Viz_Health_{tag}.png', dpi=300)
    plt.close()


def plot_global_heatmap(df, output_dir, tag):
    """Plots Layer vs Multiplier for the 'Gap' metric."""
    plt.figure(figsize=(14, 10))

    pivot = df.pivot(index='Layer', columns='Multiplier', values='Gap')
    pivot.sort_index(inplace=True)

    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm", center=0,
                cbar_kws={'label': 'Deontological Advantage (%)'}, annot_kws={"size": 8})
    
    plt.title(f'Global Steering Heatmap ({tag})', fontsize=16, weight='bold')
    plt.xlabel('Multiplier', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_dir / f'Viz_Global_Heatmap_{tag}.png', dpi=300)
    plt.close()


def plot_refusal_wall(df, output_dir, tag):
    """Plots Layer vs Multiplier for Invalid Rate."""
    plt.figure(figsize=(14, 10))
    
    pivot = df.pivot(index='Layer', columns='Multiplier', values='Invalid_Rate')
    pivot.sort_index(inplace=True)

    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100,
                cbar_kws={'label': 'Invalid Rate (%)'}, annot_kws={"size": 8})
    
    plt.title(f'Global Refusal Wall ({tag})', fontsize=16, weight='bold')
    plt.xlabel('Multiplier', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f'Viz_Global_Refusal_{tag}.png', dpi=300)
    plt.close()


# def plot_quartered_analysis(df, output_dir, tag):
#     """Generates the 4x4 Grid breakdown (Only runs if we have enough layers)."""
#     chunks = [
#         (0, 7, "Quadrant 1: Early Layers (0-7)"),
#         (8, 15, "Quadrant 2: Middle Layers (8-15)"),
#         (16, 23, "Quadrant 3: Deep Layers (16-23)"),
#         (24, 31, "Quadrant 4: Final Layers (24-31)")
#     ]
    
#     fig, axes = plt.subplots(2, 2, figsize=(20, 16))
#     axes = axes.flatten()

#     for i, (start, end, title) in enumerate(chunks):
#         chunk_df = df[(df['Layer'] >= start) & (df['Layer'] <= end)]
        
#         # S-Curve Subplot
#         sns.lineplot(data=chunk_df, x='Multiplier', y='Deon_Score', hue='Layer',
#                      palette='tab10', marker='o', ax=axes[i], legend='full')
        
#         axes[i].set_title(title, fontsize=12, weight='bold')
#         axes[i].set_ylim(-5, 105)
#         axes[i].axhline(50, color='gray', linestyle='--')

#     plt.suptitle(f'Quartered S-Curve Analysis ({tag})', fontsize=20, weight='bold')
#     plt.tight_layout()
    
#     save_path = os.path.join(output_dir, f'Viz_Quartered_{tag}.png')
#     plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()

# --- QUARTERED PLOTTING FUNCTIONS ---

CHUNKS = [
    (0, 7, "Quadrant 1: Early Layers (0-7)"),
    (8, 15, "Quadrant 2: Middle Layers (8-15)"),
    (16, 23, "Quadrant 3: Deep Layers (16-23)"),
    (24, 31, "Quadrant 4: Final Layers (24-31)")
]    

def plot_quartered_heatmap(df, value_col, title_prefix, filename_suffix, cmap, output_dir, tag, vmin=None, vmax=None, center=None, fmt=".1f"):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for i, (start, end, title) in enumerate(CHUNKS):
        chunk_df = df[(df['Layer'] >= start) & (df['Layer'] <= end)]
        
        # Hide empty subplots to prevent crashes on partial sweeps
        if chunk_df.empty:
            axes[i].set_visible(False)
            continue
            
        pivot = chunk_df.pivot(index='Layer', columns='Multiplier', values=value_col)
        pivot.sort_index(inplace=True)

        sns.heatmap(pivot, ax=axes[i], cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                    annot=True, fmt=fmt, annot_kws={"size": 10}, 
                    cbar=True if i in [1,3] else False)

        axes[i].set_title(title, fontsize=14, weight='bold')
        axes[i].set_xlabel('Multiplier')
        axes[i].set_ylabel('Layer')

    plt.suptitle(f'{title_prefix} ({tag})', fontsize=20, weight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'Viz_Quartered_{filename_suffix}_{tag}.png', dpi=300)
    plt.close()    


def plot_quartered_line(df, value_col, title_prefix, filename_suffix, output_dir, tag, hline_val=None, hline_color='gray'):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for i, (start, end, title) in enumerate(CHUNKS):
        chunk_df = df[(df['Layer'] >= start) & (df['Layer'] <= end)]
        
        # CRITICAL FIX: Hide empty subplots
        if chunk_df.empty:
            axes[i].set_visible(False)
            continue

        sns.lineplot(data=chunk_df, x='Multiplier', y=value_col, hue='Layer',
                     palette='tab10', marker='o', ax=axes[i], legend='full')

        if hline_val is not None:
            axes[i].axhline(hline_val, color=hline_color, linestyle='--', alpha=0.7)

        axes[i].set_title(title, fontsize=14, weight='bold')
        axes[i].set_ylabel(value_col)
        axes[i].set_xlabel('Multiplier')
        axes[i].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    plt.suptitle(f'{title_prefix} ({tag})', fontsize=20, weight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'Viz_Quartered_{filename_suffix}_{tag}.png', dpi=300)
    plt.close()    