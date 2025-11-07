import pandas as pd
import numpy as np
import os
import json
import time
import sys

# Import function to process 40,000,000 examples in chuncks
from src.utils import process_csv_in_chunks 

from src.scenario_builder import (
    parse_stay_scenario, 
    parse_swerve_scenario,
    generate_text_pair
)

from src.ethical_frameworks import (
    get_utilitarian_choice,
    lookup_deontological_reason_text
)


# Import paths and settings from config file
from src.config import (
    ORIGINAL_DATA_PATH,
    STEERING_PROMPT_PATH,
    STEERING_PROMPT_DIR,
    CHARACTER_COLS,
    NUM_PROMPT_PAIRS_TO_GENERATE
)

# --- Helper functions ---

def get_deontological_choice_with_reason(row, stay_info, swerve_info):
    # barrier indicated the Anti-Sacrifice principal which overides all
    try:
        if int(row['Barrier']) == 1:
            return 'stay', 'Anti-Sacrifice'
        return 'stay', 'Inaction'
    except (ValueError, KeyError, TypeError) as e:
         # print(f"   WARNING: Error in deontological classification: {e}") 
         return 'stay', 'Inaction_ErrorFallback'
REASON_CLAUSE_U = " because saving the maximum number of lives leads to the best overall outcome according to utilitarian principles."


# --- 1. Define the Row Processor Function ---
# Process each row from original dataset
def process_prompt_generation_row(row):
    """
    Process single row to checK a U-vs-D conflict,
    Returns the augmented prompt pair. Otherwise, returns None.
    """
    
    # --- Per row cleaning ---
    try:
        # Fill NaNs for columns we're about to use
        row[CHARACTER_COLS] = row[CHARACTER_COLS].fillna(0)
        row['CrossingSignal'] = row['CrossingSignal'].fillna(0)
        row['Barrier'] = int(row['Barrier'].fillna(0))
        row['DiffNumberOFCharacters'] = int(pd.to_numeric(row['DiffNumberOFCharacters'], errors='coerce').fillna(0))
    except Exception as e:
        # print(f"Skipping row for data error {e}")
        return None # Skip row if fails

    # --- Parse & Classify ---
    stay_details = parse_stay_scenario(row)
    swerve_details = parse_swerve_scenario(row, stay_details['total_count'])
    
    U_choice_action = get_utilitarian_choice(stay_details, swerve_details)
    D_choice_action, D_reason_code = get_deontological_choice_with_reason(row, stay_details, swerve_details)

    # --- Filter for U vs D Conflict ---
    if U_choice_action == 'equal' or U_choice_action == D_choice_action:
        return None # No conflict, skip this row

    # --- Generate and Return the Result ---
    text_stay, text_swerve = generate_text_pair(row, stay_details, swerve_details)
    base_text_D = text_stay if D_choice_action == 'stay' else text_swerve
    base_text_U = text_stay if U_choice_action == 'stay' else text_swerve
    reason_clause_D = lookup_deontological_reason_text(D_reason_code)
    
    prompt_D = base_text_D + reason_clause_D
    prompt_U = base_text_U + REASON_CLAUSE_U
    
    result_dict = {
        'DeontologicalPrompt': prompt_D,
        'UtilitarianPrompt': prompt_U,
        'Original_ResponseID': row['ResponseID'],
        'DeontologicalReasonCode': D_reason_code
    }
    # Return the valuable data
    return result_dict


# --- Main Function (TEST MODE VERSION) ---
def generate_steering_prompts(num_samples_to_generate=500, test_run_rows=10000):
    """
    Main function to generate and save augmented steering prompts.
    
    *** TEST MODE ***
    This version reads only the first `test_run_rows` rows from the
    original data file for a quick logic test.
    """
    
    print("="*30)
    print(f"TEST MODEL")
    print(f"Reading only the first {test_run_rows} rows of the dataset.")
    print("="*30)
    
    # --- 1. Load Data (Test Subset) ---
    load_start_time = time.time()
    print(f"Loading first {test_run_rows} rows from: {ORIGINAL_DATA_PATH}...")
    try:
        # test_run_rows = no row to sample
        df_full = pd.read_csv(ORIGINAL_DATA_PATH, nrows=test_run_rows)
        
        # Clean data
        df_full[CHARACTER_COLS] = df_full[CHARACTER_COLS].fillna(0)
        df_full['CrossingSignal'] = df_full['CrossingSignal'].fillna(0)
        df_full['Barrier'] = df_full['Barrier'].fillna(0).astype(int)
        df_full['DiffNumberOFCharacters'] = pd.to_numeric(df_full['DiffNumberOFCharacters'], errors='coerce').fillna(0).astype(int)
        
        load_end_time = time.time()
        print(f"Loaded and cleaned {len(df_full)} rows in {load_end_time - load_start_time:.2f} seconds.")
        
    except FileNotFoundError:
        print(f"File not found at '{ORIGINAL_DATA_PATH}'.")
        return
    except Exception as e:
        print(f"cleaning error: {e}")
        return

    # --- 2. Process and Generate ---
    augmented_prompt_pairs = []
    processed_count = 0
    skipped_count = 0
    total_scenarios = len(df_full)
    start_loop_time = time.time()
    
    print(f"Filtering {total_scenarios} scenarios to find U-vs-D conflicts...")
    
    # Iterate through the DataFrame
    for i, row in df_full.iterrows():
        
        # Only need 500 prompt per choice - exit once complete
        if processed_count >= num_samples_to_generate:
            print(f"\nTarget number of {num_samples_to_generate} prompt pairs reached. Stopping.")
            break

        stay_details = parse_stay_scenario(row)
        swerve_details = parse_swerve_scenario(row, stay_details['total_count'])
        
        U_choice_action = get_utilitarian_choice(stay_details, swerve_details)
        D_choice_action, D_reason_code = get_deontological_choice_with_reason(row, stay_details, swerve_details)

        if U_choice_action == 'equal' or U_choice_action == D_choice_action:
            skipped_count += 1
            continue 

        text_stay, text_swerve = generate_text_pair(row, stay_details, swerve_details)
        base_text_D = text_stay if D_choice_action == 'stay' else text_swerve
        base_text_U = text_stay if U_choice_action == 'stay' else text_swerve
        reason_clause_D = lookup_deontological_reason_text(D_reason_code)
        
        prompt_D = base_text_D + reason_clause_D
        prompt_U = base_text_U + REASON_CLAUSE_U
        
        augmented_prompt_pairs.append({
            'DeontologicalPrompt': prompt_D,
            'UtilitarianPrompt': prompt_U,
            'Original_ResponseID': row['ResponseID'],
            'DeontologicalReasonCode': D_reason_code
        })
        processed_count += 1

    end_loop_time = time.time()
    print(f"\n--- Processing Complete (Test Run) ---")
    print(f"Total time taken: {end_loop_time - start_loop_time:.2f} seconds.")
    print(f"Generated {processed_count} augmented prompt pairs.")
    print(f"Skipped {skipped_count} scenarios (no conflict).")

    # --- 3. Save to JSON ---
    if processed_count > 0:
        os.makedirs(STEERING_PROMPT_DIR, exist_ok=True)
        print(f"\nSaving generated prompts to: {STEERING_PROMPT_PATH}...")
        try:
            with open(STEERING_PROMPT_PATH, 'w', encoding='utf-8') as f:
                json.dump(augmented_prompt_pairs, f, indent=4, ensure_ascii=False)
            print(f"Saved {processed_count} pairs.")
        except Exception as e:
            print(f"ERROR: Failed to save JSON file. Error: {e}")
    else:
        print(f"No prompt pairs were generated from the first {test_run_rows} rows.")

# Main
if __name__ == "__main__":
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.config import STEERING_PROMPT_PATH, NUM_PROMPT_PAIRS_TO_GENERATE
    
    print("Running script directly (locally in TEST MODE)...")
    # Call with a specific test row limit
    generate_steering_prompts(num_samples_to_generate=NUM_PROMPT_PAIRS_TO_GENERATE, test_run_rows=10000)
    print(f"Check the output file at: {STEERING_PROMPT_PATH}")