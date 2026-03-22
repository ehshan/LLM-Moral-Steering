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
    The Master Visualisation Function.
    Takes raw aggregated results, dynamically detects the data structure (Single-Layer 
    vs Multi-Layer Profile), and routes to the correct plotting suite.
    
    Args:
        df (pd.DataFrame): The aggregated results dataframe.
        output_dir (Path): Where to save the generated images.
        experiment_tag (str): Label for the saved files.
    """
    print(f"Generating Analysis Report for: {experiment_tag}")
    
    # Ensure output_dir is a Path object for cross-platform compatibility
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-Processing: Calculate the "Gap" with MASKING
    # If Invalid_Rate is > 15%, set Gap to NaN to draw a blank square instead of "Neutral 0".
    df['Gap'] = np.where(
        df['Invalid_Rate'] > 15, 
        np.nan, 
        df['Deon_Score'] - df['Util_Score']
    )
    
    # --- Dynamic Data Sniffing ---
    if 'Profile_Name' in df.columns and 'Sweep_Multiplier' in df.columns:
        entity_col = 'Profile_Name'
        is_profile_mode = True
    elif 'Layer' in df.columns and 'Multiplier' in df.columns:
        entity_col = 'Layer'
        is_profile_mode = False
    else:
        raise KeyError("Dataframe must contain ('Profile_Name', 'Sweep_Multiplier') or ('Layer', 'Multiplier').")

    unique_entities = sorted(df[entity_col].unique())
    print(f"   -> Detected {len(unique_entities)} unique {entity_col}(s).")
    
    # --- Structural Logic ---
    if is_profile_mode:
        # Profiles always generate heatmaps, but never quartered grids.
        generate_heatmaps = True
        generate_quartered = False
    else:
        # Baseline layers generate heatmaps and grids only if testing 20+ layers.
        is_full_sweep = len(unique_entities) >= 20
        generate_heatmaps = is_full_sweep
        generate_quartered = is_full_sweep

    # --- Dynamic Subsetting for Line Plots ---
    # Prevents spaghetti graphs on massive layer sweeps. Skipped for profiles.
    if not is_profile_mode and len(unique_entities) > 6:
        idx = np.round(np.linspace(0, len(unique_entities) - 1, 4)).astype(int)
        target_entities = [unique_entities[i] for i in idx]
        print(f"   -> Subsetting line plots to representative layers: {target_entities}")
        line_df = df[df[entity_col].isin(target_entities)]
    else:
        line_df = df

    # --- Standard Plots (Run for all experiments) ---
    plot_s_curve(line_df, output_dir, experiment_tag)
    # plot_health_line(line_df, output_dir, experiment_tag) # (Requires similar dynamic update)
    
    # --- Conditional Plots ---
    if generate_heatmaps:
        print("   -> Mode: Comprehensive (Generating Heatmaps)")
        plot_global_heatmap(df, output_dir, experiment_tag)
        plot_refusal_wall(df, output_dir, experiment_tag)
        
    if generate_quartered:
        print("   -> Mode: Full Sweep (Generating Quartered Grids)")
        plot_quartered_heatmap(df, 'Gap', 'Steering Effectiveness', 'Heatmap_Gap', 
                               cmap='coolwarm', center=0, fmt=".1f", output_dir=output_dir, tag=experiment_tag)
        plot_quartered_heatmap(df, 'Invalid_Rate', 'Refusal Wall', 'Heatmap_Invalid', 
                               cmap='Reds', vmin=0, vmax=100, fmt=".0f", output_dir=output_dir, tag=experiment_tag)
        plot_quartered_line(df, 'Deon_Score', 'S-Curve Response', 'SCurve', 
                            hline_val=50, output_dir=output_dir, tag=experiment_tag)
        plot_quartered_line(df, 'Invalid_Rate', 'Model Health', 'Health', 
                            hline_val=10, hline_color='red', output_dir=output_dir, tag=experiment_tag)
    elif not is_profile_mode:
        print("   -> Mode: Precision Strike (Skipping Quartered Grids)")
        
    print(f"Visualisation Complete. Saved to: {output_dir}")


# --- INDIVIDUAL PLOTTING FUNCTIONS ---

def plot_s_curve(df, output_dir, tag):
    """
    Plots the S-Curve (Multiplier vs Deontological Score).
    Dynamically maps axes based on whether data is Layer-based or Profile-based.
    """
    plt.figure(figsize=(10, 6))

    # Dynamic Column Sniffing
    if 'Profile_Name' in df.columns:
        entity_col = 'Profile_Name'
        x_col = 'Sweep_Multiplier'
        legend_title = 'Intervention Profile'
        x_label = 'Conceptual Sweep Multiplier'
    else:
        entity_col = 'Layer'
        x_col = 'Multiplier'
        legend_title = 'Layer Index'
        x_label = 'Steering Multiplier'

    # Plotting Logic
    if df[entity_col].nunique() > 1:
        sns.lineplot(data=df, x=x_col, y='Deon_Score', hue=entity_col, 
                     palette='viridis', style=entity_col, markers=True, dashes=False, linewidth=2.5)
        plt.legend(title=legend_title)
    else:
        sns.lineplot(data=df, x=x_col, y='Deon_Score', marker='o', linewidth=3, color='blue')
        
    # Add Reference Line (50% = Neutral)
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5, label="Neutral (50%)")

    plt.title(f'Steering Response S-Curve ({tag})', fontsize=14, weight='bold')
    plt.ylabel('Deontological Choice (%)')
    plt.xlabel(x_label)
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
    """
    Plots the target entity vs multiplier for the 'Gap' metric.
    Dynamically maps axes based on whether data is Layer-based or Profile-based.
    """
    plt.figure(figsize=(14, 10))

    # Dynamic Column Sniffing
    if 'Profile_Name' in df.columns:
        entity_col = 'Profile_Name'
        x_col = 'Sweep_Multiplier'
        y_label = 'Intervention Profile'
        x_label = 'Conceptual Sweep Multiplier'
    else:
        entity_col = 'Layer'
        x_col = 'Multiplier'
        y_label = 'Layer Index'
        x_label = 'Steering Multiplier'

    # Pivot the dataframe using the dynamically assigned columns
    pivot = df.pivot(index=entity_col, columns=x_col, values='Gap')
    pivot.sort_index(inplace=True)

    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm", center=0,
                cbar_kws={'label': 'Deontological Advantage (%)'}, annot_kws={"size": 8})
    
    plt.title(f'Global Steering Heatmap ({tag})', fontsize=16, weight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.tight_layout()

    plt.savefig(output_dir / f'Viz_Global_Heatmap_{tag}.png', dpi=300)
    plt.close()


def plot_refusal_wall(df, output_dir, tag):
    """
    Plots the target entity vs multiplier for Invalid Rate.
    Dynamically maps axes based on whether data is Layer-based or Profile-based.
    """
    plt.figure(figsize=(14, 10))
    
    # Dynamic Column Sniffing
    if 'Profile_Name' in df.columns:
        entity_col = 'Profile_Name'
        x_col = 'Sweep_Multiplier'
        y_label = 'Intervention Profile'
        x_label = 'Conceptual Sweep Multiplier'
    else:
        entity_col = 'Layer'
        x_col = 'Multiplier'
        y_label = 'Layer Index'
        x_label = 'Steering Multiplier'

    # Pivot the dataframe using the dynamically assigned columns
    pivot = df.pivot(index=entity_col, columns=x_col, values='Invalid_Rate')
    pivot.sort_index(inplace=True)

    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100,
                cbar_kws={'label': 'Invalid Rate (%)'}, annot_kws={"size": 8})
    
    plt.title(f'Global Refusal Wall ({tag})', fontsize=16, weight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
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