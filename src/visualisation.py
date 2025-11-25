import matplotlib.pyplot as plt

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