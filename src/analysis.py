"""
Data Analysis & Aggregation Module
Handles the processing of raw evaluation data, merging of AI Judge results, 
and mathematical extraction of steering metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import List, Optional
from typing import Optional


def aggregate_sweep_results(
    raw_files: List[str], 
    method: str, 
    timestamp: str, 
    output_dir: Path
) -> Optional[pd.DataFrame]:
    """
    Merges raw generation logs with AI Judge progress files to calculate final 
    percentage scores (Deontological, Utilitarian, Invalid) per layer and multiplier.

    Args:
        raw_files (List[str]): List of filenames for the raw generation chunks.
        method (str): The judging method used (e.g., 'ai' or 'strict').
        timestamp (str): The timestamp string identifying the run.
        output_dir (Path): The directory where results are stored and saved.

    Returns:
        pd.DataFrame or None: The aggregated master dataframe, or None if failure.
    """
    print(f"--- Starting Master Aggregation ({method.upper()} Judge) ---")
    all_summaries = []

    for raw_name in raw_files:
        raw_path = output_dir / raw_name
        
        # Resolve the matching progress file name 
        # (Handles legacy double-tagging issues like 'ai_ai')
        prog_name_standard = raw_name.replace("raw_responses_", f"progress_{method}_")
        prog_name_double = raw_name.replace("raw_responses_", f"progress_{method}_{method}_")
        
        prog_path = output_dir / prog_name_standard
        if not prog_path.exists():
            prog_path = output_dir / prog_name_double

        if raw_path.exists() and prog_path.exists():
            print(f"[+] Processing Chunk: {raw_name}")
            df_raw = pd.read_csv(raw_path)
            df_prog = pd.read_csv(prog_path)
            
            # Map the AI judgments back to the original rows via index/Row_ID
            df_raw['Result'] = df_raw.index.map(df_prog.set_index('Row_ID')['Result'])
            
            # Calculate the percentages: group by Layer/Multiplier and normalize
            summary = (
                df_raw.groupby(['Layer', 'Multiplier'])['Result']
                .value_counts(normalize=True)
                .unstack(fill_value=0) * 100
            )
            
            # Ensure standard categorical columns exist even if no errors occurred in this chunk
            for col in ['Deontological', 'Utilitarian', 'INVALID', 'API_ERROR']:
                if col not in summary.columns: 
                    summary[col] = 0.0
                    
            # Standardise column names for downstream visualiser compatibility
            summary = summary.rename(columns={
                'Deontological': 'Deon_Score', 
                'Utilitarian': 'Util_Score', 
                'INVALID': 'Invalid_Rate'
            })
            
            # Fold API_ERRORs into the Invalid_Rate to ensure clean masking on heatmaps
            if 'API_ERROR' in summary.columns:
                summary['Invalid_Rate'] += summary['API_ERROR']
                summary = summary.drop(columns=['API_ERROR'])
                
            summary.reset_index(inplace=True)
            all_summaries.append(summary)
        else:
            print(f"Warning: Missing raw or progress file for {raw_name}. Skipping.")

    # Final Merge & Save
    if all_summaries:
        master_df = pd.concat(all_summaries, ignore_index=True)
        expected_rows = master_df['Layer'].nunique() * master_df['Multiplier'].nunique()
        
        print(f"\nCreated Master DataFrame with {len(master_df)} rows. (Expected: {expected_rows})")
        
        # Save the aggregated master file
        master_filename = f"strength_sweep_results_0-31_AGGREGATED_{method}.csv"
        master_filepath = output_dir / master_filename
        master_df.to_csv(master_filepath, index=False)
        print(f"Saved to: {master_filepath}")
        
        return master_df
    else:
        print("Error: Could not process any chunks. Check file paths and names.")
        return None
    

def generate_metrics_atlas(
    df: pd.DataFrame, 
    output_dir: Path, 
    tag: str, 
    invalid_threshold: float = 15.0
) -> Optional[pd.DataFrame]:
    """
    Scans the aggregated dataframe to calculate maximum safe semantic shifts,
    total control bandwidth, and the precise boundaries of refusal/collapse walls.
    Exports the results as a CSV and a formatted PNG table.

    Args:
        df (pd.DataFrame): The aggregated master dataframe.
        output_dir (Path): The directory to save the output files.
        tag (str): The experiment identifier for file naming.
        invalid_threshold (float): The maximum allowed invalid rate before a 
                                   multiplier is considered a "wall".

    Returns:
        pd.DataFrame or None: The metrics atlas dataframe, or None if it fails.
    """
    print(f"--- Generating Layer Metrics Atlas ({tag}) ---")
    
    # Ensure standard column naming conventions before processing
    rename_map = {
        'Deontological': 'Deon_Score', 
        'Utilitarian': 'Util_Score', 
        'INVALID': 'Invalid_Rate'
    }
    df = df.rename(columns=rename_map)
    
    # Validate required columns exist
    required_cols = ['Layer', 'Multiplier', 'Deon_Score', 'Invalid_Rate']
    if not all(col in df.columns for col in required_cols):
        print("Error: Missing required columns for atlas generation. Check dataframe.")
        return None

    metrics_data = []
    
    # Mathematical extraction per layer
    for layer in sorted(df['Layer'].unique()):
        layer_df = df[df['Layer'] == layer]
        
        # Isolate the baseline (0.0 multiplier)
        baseline_row = layer_df[layer_df['Multiplier'] == 0.0]
        if baseline_row.empty:
            continue
        baseline_deon = baseline_row['Deon_Score'].values[0]
        
        # Filter strictly for valid responses below the collapse threshold
        valid_df = layer_df[layer_df['Invalid_Rate'] <= invalid_threshold]
        
        # Handle entirely broken layers
        if valid_df.empty:
            metrics_data.append({
                'Layer': int(layer), 
                'Baseline (Deon)': f"{baseline_deon:.1f}%",
                'Max Deon Shift': "N/A", 
                'Max Util Shift': "N/A", 
                'Total Bandwidth': "0.0%", 
                'Walls (- | +)': "[BROKEN]"
            })
            continue
            
        # Calculate maximum shifts safely within the valid corridor
        max_deon = valid_df['Deon_Score'].max()
        min_deon = valid_df['Deon_Score'].min()
        
        peak_shift_deon = max_deon - baseline_deon
        peak_shift_util = min_deon - baseline_deon  # Negative indicates Util shift
        bandwidth = max_deon - min_deon
        
        # Locate the architectural breakdown boundaries (Walls)
        invalid_df = layer_df[layer_df['Invalid_Rate'] > invalid_threshold]
        pos_invalids = invalid_df[invalid_df['Multiplier'] > 0]['Multiplier']
        neg_invalids = invalid_df[invalid_df['Multiplier'] < 0]['Multiplier']
        
        pos_wall = f"+{pos_invalids.min()}" if not pos_invalids.empty else "None"
        neg_wall = f"{neg_invalids.max()}" if not neg_invalids.empty else "None"
        
        metrics_data.append({
            'Layer': int(layer),
            'Baseline (Deon)': f"{baseline_deon:.1f}%",
            'Max Deon Shift': f"+{peak_shift_deon:.1f}%" if peak_shift_deon > 0 else f"{peak_shift_deon:.1f}%",
            'Max Util Shift': f"{peak_shift_util:.1f}%",
            'Total Bandwidth': f"{bandwidth:.1f}%",
            'Walls (- | +)': f"[{neg_wall} | {pos_wall}]"
        })
        
    metrics_df = pd.DataFrame(metrics_data)
    
    # Console output for immediate review
    print("\nMetrics Extraction Complete:")
    print(metrics_df.to_string(index=False))
    
    # Deliverable 1: Save as raw CSV data
    csv_out_path = output_dir / f"Layer_Metrics_Atlas_{tag}.csv"
    metrics_df.to_csv(csv_out_path, index=False)
    print(f"\nCSV saved to: {csv_out_path}")
    
    # Deliverable 2: Render as a formatted PNG table for documentation
    png_out_path = output_dir / f"Layer_Metrics_Atlas_{tag}.png"
    
    # Dynamically size the figure based on the number of rows
    fig, ax = plt.subplots(figsize=(10, len(metrics_df) * 0.3 + 1.5))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(
        cellText=metrics_df.values, 
        colLabels=metrics_df.columns, 
        cellLoc='center', 
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Apply styling and alternating row colours for readability
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4C72B0')
        elif row % 2 == 0:
            cell.set_facecolor('#F3F6F9')
            
    plt.title(f"Layer Control Atlas: Steering Metrics ({tag})", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(png_out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PNG Table saved to: {png_out_path}")
    return metrics_df


def compare_parsers(
    strict_csv_path: Path, 
    ai_csv_path: Path, 
    output_dir: Path, 
    tag: str
) -> Optional[pd.DataFrame]:
    """
    Compares the Invalid Rate (Refusal Wall) between the Strict Regex Parser 
    and the AI Judge, specifically targeting Layer 6 (the Safety Valve).
    
    This visualises the core hypothesis: The AI judge rescues valid moral 
    reasoning data that the strict regex parser throws away due to minor 
    formatting degradation.

    Args:
        strict_csv_path (Path): Path to the aggregated CSV from the strict sweep.
        ai_csv_path (Path): Path to the aggregated CSV from the AI judge sweep.
        output_dir (Path): Directory to save the output plot.
        tag (str): Experiment identifier for the saved filename.

    Returns:
        pd.DataFrame or None: The combined dataframe for Layer 6, or None if files are missing.
    """
    print(f"--- Starting Comparison Analysis: Strict Parser vs AI Judge ({tag}) ---")
    
    if not strict_csv_path.exists() or not ai_csv_path.exists():
        print("Error: Missing one or both required CSV files for comparison.")
        print("You must run both a 'strict' sweep and an 'ai' sweep first.")
        return None

    # Load both datasets
    df_strict = pd.read_csv(strict_csv_path)
    df_judge = pd.read_csv(ai_csv_path)

    # Standardise column names just in case older strict sweeps used legacy names
    rename_map = {'INVALID': 'Invalid_Rate'}
    df_strict = df_strict.rename(columns=rename_map)
    df_judge = df_judge.rename(columns=rename_map)

    # Label the datasets
    df_strict['Method'] = 'Strict Parser'
    df_judge['Method'] = 'AI Judge'

    # Combine and isolate Layer 6 (The "Moral Pivot" / Safety Valve)
    df_combined = pd.concat([df_strict, df_judge], ignore_index=True)
    layer_6_df = df_combined[df_combined['Layer'] == 6]

    if layer_6_df.empty:
        print("Error: No data found for Layer 6 in the provided files.")
        return None

    # Generate the comparison line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=layer_6_df, 
        x='Multiplier', 
        y='Invalid_Rate', 
        hue='Method',
        style='Method', 
        markers=True, 
        linewidth=2.5
    )

    # Add the usability threshold line
    plt.axhline(10, color='red', linestyle='--', label="Usable Threshold (10%)")
    
    plt.title("Refusal Wall: Strict Parser vs. AI Judge (Layer 6)", fontsize=14, weight='bold')
    plt.ylabel("Invalid Rate (%)")
    plt.xlabel("Steering Multiplier")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    save_path = output_dir / f"Viz_Comparison_Layer6_{tag}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Comparison plot saved to: {save_path}")
    print("Analysis: If the 'AI Judge' line is lower (further to the right), it proves")
    print("that many 'Refusals' flagged by the strict parser were just formatting errors.")
    
    return layer_6_df