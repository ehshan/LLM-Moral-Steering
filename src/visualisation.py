import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

def plot_layer_sweep(results_log):
    """
    Visualizes the steering effect across layers.
    X-axis: Layer Number
    Y-axis (Left): % Deontological Choice (Effectiveness)
    Y-axis (Right): % Invalid Responses (Safety)
    """
    layers = [r['Layer'] for r in results_log]
    pos_scores = [r['Pos_Deon'] for r in results_log]
    neg_scores = [r['Neg_Deon'] for r in results_log]
    invalid_scores = [r['Invalid_Avg'] for r in results_log]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Effectiveness (The Gap)
    ax1.set_xlabel('Layer Number', fontsize=12)
    ax1.set_ylabel('% Deontological Choice', color='tab:blue', fontsize=12)
    ax1.plot(layers, pos_scores, label='+1.0 (Push Deon)', color='tab:blue', marker='o', linewidth=2)
    ax1.plot(layers, neg_scores, label='-1.0 (Push Util)', color='tab:cyan', marker='o', linestyle='--', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)

    # Plot Safety (Invalid Rate) on Secondary Y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('% Invalid Responses', color='tab:red', fontsize=12)
    ax2.bar(layers, invalid_scores, color='tab:red', alpha=0.2, label='Invalid %', width=0.6)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 50) # Scale this depending on how bad the invalid rate gets

    # Title and Legend
    plt.title('Steering Layer Sweep: Effectiveness vs. Safety', fontsize=14, pad=20)
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_strength_heatmap(df_results):
    """
    Plots a heatmap: Layer vs. Multiplier with color representing Deon Score.
    """
    # Pivot data for heatmap format
    heatmap_data = df_results.pivot(index="Layer", columns="Multiplier", values="Deon_Score")
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="Blues", cbar_kws={'label': '% Deontological'})
    plt.title("Steering Effectiveness Heatmap (Layer vs. Strength)")
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