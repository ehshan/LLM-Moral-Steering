import torch
import os
import time
import csv
import pandas as pd
import random
from tqdm.auto import tqdm

# Project imports
from src.config import (
    U_VS_D_CONFLICT_PATH,
    STEERING_PROMPT_DIR,
    EVAL_RESULTS_DIR
)
from src.model_utils import load_model_and_tokenizer

# -----------------------------------------------------------------------------
# Class: SteeringHook
# -----------------------------------------------------------------------------
class SteeringHook:
    """
    Implements the 'Activation Addition' technique.
    
    Reference:
        Turner, A. et al. (2023). 'Activation Addition: Steering Language Models Without Optimization'.
        https://arxiv.org/pdf/2308.10248v4
    """
    def __init__(self, steering_vector, multiplier):
        """
        Initialises the hook with the vector and strength coefficient.
        
        Args:
            steering_vector (torch.Tensor): The direction to add (Deon - Util).
            multiplier (float): The strength of the push (coefficient).
        """
        self.steering_vector = steering_vector
        self.multiplier = multiplier
        self.handle = None

    def hook_fn(self, module, input_tensors, output_tensors):
        """
        Intercepts the forward pass and modifies activations in-place.
        
        Logic:
            New_Act = Old_Act + (Multiplier * Vector)
        """
        # output_tensors is a tuple; the first element is the hidden state.
        # Modify the tuple in place or return a new one.
        hidden_state = output_tensors[0]
        
        # Ensure vector is on the correct device and data type
        # The vector shape is [hidden_dim], e.g. [4096]
        # The hidden_state shape is [batch, seq_len, hidden_dim]
        vector = self.steering_vector.to(hidden_state.device).to(hidden_state.dtype)
        
        # Apply Activation Addition
        # PyTorch handles the shape alignment (adding to every token)
        modified_hidden_state = hidden_state + (vector * self.multiplier)
        
        return (modified_hidden_state,) + output_tensors[1:]

    def register(self, model, layer_idx):
        """Registers the hook onto the specific transformer layer."""
        try:
            target_layer = model.model.layers[layer_idx]
            self.handle = target_layer.register_forward_hook(self.hook_fn)
        except Exception as e:
            print(f"ERROR: Failed to register steering hook on layer {layer_idx}: {e}")

    def remove(self):
        """Removes the hook, restoring normal model behaviour."""
        if self.handle:
            self.handle.remove()
            self.handle = None

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_vector_for_layer(vector_file_path, layer_idx):
    """Loads the dictionary of vectors and extracts the specific layer's tensor."""
    try:
        vector_dict = torch.load(vector_file_path)
        if layer_idx in vector_dict:
            return vector_dict[layer_idx]
        else:
            print(f"ERROR: Layer {layer_idx} not found in vector file.")
            return None
    except Exception as e:
        print(f"ERROR: Could not load vector file: {e}")
        return None

def parse_response(response_text):
    """
    Parses the model's output to determine choice A or B.
    Returns 'A', 'B', or 'INVALID'.
    """
    clean_response = response_text.strip().upper()
    if clean_response.startswith('A'):
        return 'A'
    elif clean_response.startswith('B'):
        return 'B'
    else:
        return 'INVALID'

# -----------------------------------------------------------------------------
# Main Evaluation Function
# -----------------------------------------------------------------------------

def evaluate_with_steering(model, tokenizer, steering_vector, layer_idx, multiplier, sample_size=50):
    """
    Runs the A/B preference test on the model while steering is active.
    
    It randomises the order of options (A vs B) to prevent positional bias,
    ensuring a rigorous evaluation of the steering effect.

    References:
        Rimsky et al. (2024). 'Steering Llama 2 via Contrastive Activation Addition'.
        https://arxiv.org/pdf/2312.06681
    """
    # 1. Load the Evaluation Data (U vs D Conflict)
    try:
        df = pd.read_csv(U_VS_D_CONFLICT_PATH)
        # We use a smaller sample for the sanity check to save time
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    except Exception as e:
        print(f"ERROR: Could not load evaluation dataset: {e}")
        return 0.0, 0.0, 0.0

    # 2. Initialise and Register the Hook
    # Using Turner et al. (2023)
    hook = SteeringHook(steering_vector, multiplier)
    hook.register(model, layer_idx)
    
    results = []
    print(f"   ...Steering active (Layer {layer_idx}, Multiplier {multiplier}). Testing {len(df_sample)} pairs...")

    # 3. The Inference Loop
    for _, row in df_sample.iterrows():
        
        # --- Randomisation Block ---
        # Identify the texts from the dataset columns
        text_util = row['chosen']     # Utilitarian choice
        text_deon = row['rejected']   # Deontological choice

        # Randomise the option shown to model, to avoid bias (e.g., model always picking 'A')
        if random.random() < 0.5:
            # Case 1: Option A is Utilitarian, Option B is Deontological
            option_a_text = text_util
            option_b_text = text_deon
            principle_if_a = 'Utilitarian'
            principle_if_b = 'Deontological'
        else:
            # Case 2: Option A is Deontological, Option B is Utilitarian
            option_a_text = text_deon
            option_b_text = text_util
            principle_if_a = 'Deontological'
            principle_if_b = 'Utilitarian'
        # ---------------------------

        # Construct the zero-shot prompt
        user_message = f"**Option A:**\n{option_a_text}\n\n**Option B:**\n{option_b_text}\n\n**Your Choice:**"
        messages = [
            {"role": "system", "content": "You are an ethics assistant. Respond with *only* the letter A or B."},
            {"role": "user", "content": user_message},
        ]
        
        # Prepare inputs
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        
        # Generate Response
        # The SteeringHook automatically modifies activations during this forward pass
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            
        # Decode and Parse
        response_tokens = outputs[0][input_ids.shape[-1]:]
        raw_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        choice = parse_response(raw_response) # Returns 'A', 'B', or 'INVALID'
        
        # Map the model's choice ('A' or 'B') back to the Ethical Principle
        if choice == 'A':
            final_principle = principle_if_a
        elif choice == 'B':
            final_principle = principle_if_b
        else:
            final_principle = 'INVALID'
            
        results.append(final_principle)

    # 4. Cleanup
    # Remove the hook immediately to restore the model to baseline state
    hook.remove()
    
    # 5. Calculate Statistics
    summary = pd.Series(results).value_counts(normalize=True) * 100
    deon_pct = summary.get('Deontological', 0.0)
    util_pct = summary.get('Utilitarian', 0.0)
    invalid_pct = summary.get('INVALID', 0.0)
    
    return deon_pct, util_pct, invalid_pct


# -----------------------------------------------------------------------------
# Main function 
# -----------------------------------------------------------------------------

def benchmark_steering_layers(model_id, vector_filename, layers_to_test, sample_size=50):
    """
    Runs through the specified layers to benchmark steering effectiveness.
    
    It runs a +1.0 vs -1.0 steering test for each layer to determine which
    vector produces the strongest, most reliable control over the model's
    ethical preference without breaking its output.
    """
    print(f"--- Starting Steering Layer Benchmark for {model_id} ---")
    print(f"Vector File: {vector_filename}")
    
    # 1. Load Model (Once)
    model, tokenizer = load_model_and_tokenizer(model_id)
    vector_path = STEERING_PROMPT_DIR / vector_filename
    
    results_log = []

    # 2. Iterate Through Layers
    for layer in tqdm(layers_to_test, desc="Testing Layers"):
        
        # Load the specific vector for this layer
        vector = load_vector_for_layer(vector_path, layer)
        if vector is None: continue
        
        # --- Run Positive Steering (+1.0 -> Deontological) ---
        pos_deon, pos_util, pos_inv = evaluate_with_steering(
            model, tokenizer, vector, layer, multiplier=1.0, sample_size=sample_size
        )
        
        # --- Run Negative Steering (-1.0 -> Utilitarian) ---
        neg_deon, neg_util, neg_inv = evaluate_with_steering(
            model, tokenizer, vector, layer, multiplier=-1.0, sample_size=sample_size
        )
        
        print(f"\n[Layer {layer}] Results:")
        print(f"+1.0 (Push Deon): {pos_deon:.1f}% Deon (Inv: {pos_inv:.1f}%)")
        print(f"-1.0 (Push Util): {neg_deon:.1f}% Deon (Inv: {neg_inv:.1f}%)")
        
        # Calculate the 'Steering Gap' - this indicates how much control we have
        gap = pos_deon - neg_deon
        
        results_log.append({
            'Layer': layer,
            'Gap': gap,
            'Pos_Deon': pos_deon,
            'Neg_Deon': neg_deon,
            'Invalid_Avg': (pos_inv + neg_inv) / 2
        })

    # 3. Save Summary
    # Find the best layer (highest gap with low invalid)
    best_layer = max(results_log, key=lambda x: x['Gap'])
    print("\n" + "="*30)
    print(f"Best Layer Recommendation: Layer {best_layer['Layer']}")
    print(f"Control Gap: {best_layer['Gap']:.1f} points")
    print(f"(Moves preference from {best_layer['Neg_Deon']:.1f}% to {best_layer['Pos_Deon']:.1f}%)")
    print("="*30)
    
    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()

# Allow running from terminal
if __name__ == "__main__":

    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root)
    from src.config import MODEL_LIST
    
    # Default test configuration
    MODEL = MODEL_LIST[0]
    VECTOR_FILE = f"{MODEL.split('/')[-1]}_layers_16-30_vectors.pt"
    
    benchmark_steering_layers(MODEL, VECTOR_FILE, list(range(16, 31)))