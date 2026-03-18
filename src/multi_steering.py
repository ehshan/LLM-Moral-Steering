import random
import pandas as pd
import torch
from tqdm.auto import tqdm
from typing import List, Dict, Any

from src.config import U_VS_D_CONFLICT_PATH, EVAL_RESULTS_DIR
from src.steering_core import load_vector_for_layer, execute_steered_generation

def run_composite_sweep(
    model, 
    tokenizer, 
    vector_file_path: str, 
    profiles: List[Dict[str, Any]], 
    sample_size: int = 500, 
    experiment_tag: str = "composite_untagged"
) -> str:
    """
    Executes a multi-layer composite steering sweep based on predefined profiles.
    
    This function isolates the specific variables to be clamped (fixed) and swept 
    (variable), dynamically building the configuration dictionary for each forward 
    pass. It outputs a flattened CSV format to maintain compatibility with the 
    existing downstream evaluation pipeline.

    Args:
        model (PreTrainedModel): The active language model.
        tokenizer (PreTrainedTokenizer): The corresponding tokenizer.
        vector_file_path (str): The file path to the saved PyTorch (.pt) vectors.
        profiles (list): A list of dictionaries defining the specific layer interventions.
                         Format: {
                             "profile_name": str,
                             "fixed_layers": {layer_idx: multiplier, ...},
                             "sweep_layers": [layer_idx, ...],
                             "sweep_multipliers": [float, ...]
                         }
        sample_size (int, optional): The number of prompts to evaluate. Defaults to 500.
        experiment_tag (str, optional): An identifier for the output file. 

    Returns:
        str: The file path to the saved CSV containing the raw generated responses.
    """
    print(f"\n--- Starting Composite Steering Execution [Tag: {experiment_tag}] ---")
    
    # 1. Load the Evaluation Data
    try:
        df = pd.read_csv(U_VS_D_CONFLICT_PATH)
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    except Exception as e:
        print(f"Error loading evaluation dataset: {e}")
        return ""

    # 2. Pre-load all required vectors into CPU memory to prevent redundant disk I/O
    print("Loading steering vectors into memory...")
    vector_dict_cache = torch.load(vector_file_path, map_location="cpu")
    
    raw_data = []

    # 3. Iterate through the user-defined intervention profiles
    for profile in profiles:
        profile_name = profile.get("profile_name", "Unnamed_Profile")
        fixed_layers = profile.get("fixed_layers", {})
        sweep_layers = profile.get("sweep_layers", [])
        sweep_multipliers = profile.get("sweep_multipliers", [])
        
        print(f"\nExecuting Profile: {profile_name}")
        
        # 4. Iterate over the variable layers defined in the profile
        for s_layer in sweep_layers:
            if s_layer not in vector_dict_cache:
                print(f"Warning: Sweep layer {s_layer} not found in vector file. Skipping.")
                continue
                
            sweep_vector = vector_dict_cache[s_layer]
            
            # 5. Iterate over the variable multipliers
            for s_mult in sweep_multipliers:
                desc = f"Sweep L{s_layer} @ {s_mult} | Fixed: {fixed_layers}"
                
                # Dynamically build the configuration dictionary for this specific sweep step
                # Format required by execute_steered_generation: {layer_idx: (vector_tensor, multiplier)}
                layer_configs = {}
                
                # Append the fixed configurations
                for f_layer, f_mult in fixed_layers.items():
                    if f_layer in vector_dict_cache:
                        layer_configs[f_layer] = (vector_dict_cache[f_layer], f_mult)
                    else:
                        print(f"Warning: Fixed layer {f_layer} not found. Skipping this clamp.")
                        
                # Append the active sweep configuration
                layer_configs[s_layer] = (sweep_vector, s_mult)
                
                # 6. Execute the generation loop across the prompt dataset
                for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc=desc, leave=False):
                    
                    text_util = row['chosen']
                    text_deon = row['rejected']
                    
                    # Randomise A/B positioning to mitigate positional bias
                    if random.random() < 0.5:
                        option_a, option_b = text_util, text_deon
                        target_a, target_b = 'Utilitarian', 'Deontological'
                    else:
                        option_a, option_b = text_deon, text_util
                        target_a, target_b = 'Deontological', 'Utilitarian'

                    user_message = f"**Option A:**\n{option_a}\n\n**Option B:**\n{option_b}\n\n**Your Choice:**"
                    
                    # Execute the latent intervention and generation
                    response = execute_steered_generation(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=user_message,
                        layer_configs=layer_configs,
                        max_new_tokens=10
                    )
                    
                    # Append results to the flattened data structure
                    raw_data.append({
                        'Profile_Name': profile_name,
                        'Fixed_Layers': str(fixed_layers),  # Stored as string to prevent pandas grouping errors
                        'Sweep_Layer': s_layer,
                        'Sweep_Multiplier': s_mult,
                        'Prompt': user_message,             # Retained for downstream AI evaluation
                        'Response': response,
                        'Principle_A': target_a,
                        'Principle_B': target_b
                    })

    # 7. Save to CSV
    df_raw = pd.DataFrame(raw_data)
    filename = f"raw_responses_composite_{experiment_tag}.csv"
    
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = EVAL_RESULTS_DIR / filename
    
    df_raw.to_csv(save_path, index=False)
    print(f"\nExecution Complete. Saved {len(df_raw)} rows to: {filename}")
    
    return save_path