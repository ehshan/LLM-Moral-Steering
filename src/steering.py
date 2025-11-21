import torch
import json
import os
import time
from tqdm.auto import tqdm # For output progress bar

# Import from project files
from src.config import STEERING_PROMPT_PATH, STEERING_PROMPT_DIR, MODEL_LIST
from src.model_utils import load_model_and_tokenizer

# --- 1. Multi-Layer Activation Hook ---

class MultiLayerActivationHook:
    """
    A smart hook that can capture activations from multiple layers at once.
    """
    def __init__(self):
        # Stores the activations for the *current* forward pass
        # Format: {layer_index: tensor}
        self.layer_activations = {} 
        self.handles = []

    def get_hook_fn(self, layer_idx):
        """
        Creates a specific hook function for a specific layer index.
        This closure ensures the hook knows which layer it is attached to.
        """
        def hook_fn(module, input_tensors, output_tensors):
            # The output is a tuple (hidden_state, ...)
            hidden_state = output_tensors[0]
            
            # --- Safety Check for Tensor Dimensions ---
            if hidden_state.dim() == 3:
                # [batch, seq, dim] -> Take last token
                activation = hidden_state[0, -1, :].detach().cpu()
            elif hidden_state.dim() == 2:
                # [seq, dim] -> Take last token
                activation = hidden_state[-1, :].detach().cpu()
            else:
                print(f"ERROR: Unexpected shape at layer {layer_idx}: {hidden_state.shape}")
                activation = None
                
            # Save to our dictionary using the layer index as the key
            self.layer_activations[layer_idx] = activation
            
        return hook_fn

    def register(self, model, target_layers):
        """
        Registers hooks on all requested layers.
        Args:
            target_layers (list): List of integers (e.g., [16, 17, 18])
        """
        for layer_idx in target_layers:
            try:
                # Create the specific hook function for this layer
                hook_fn = self.get_hook_fn(layer_idx)
                
                # Register it
                target_module = model.model.layers[layer_idx]
                handle = target_module.register_forward_hook(hook_fn)
                
                self.handles.append(handle)
            except Exception as e:
                print(f"ERROR: Failed to register hook on layer {layer_idx}. Error: {e}")

        print(f"[+] Registered hooks on {len(target_layers)} layers: {target_layers}")

    def remove(self):
        """Removes all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.layer_activations = {}
        print("[+] All hooks removed.")


# --- 2. Core Logic Functions ---

def unzip_steering_prompts():
    """Loads JSON and returns two lists of prompts."""
    print(f"Loading steering prompts from: {STEERING_PROMPT_PATH}")
    try:
        with open(STEERING_PROMPT_PATH, 'r', encoding='utf-8') as f:
            prompts_json = json.load(f)
        
        positive_prompts = [item['DeontologicalPrompt'] for item in prompts_json]
        negative_prompts = [item['UtilitarianPrompt'] for item in prompts_json]
        
        print(f"Loaded {len(positive_prompts)} pos / {len(negative_prompts)} neg prompts.")
        return positive_prompts, negative_prompts
    except Exception as e:
        print(f"ERROR: Could not read prompts. Error: {e}")
        return None, None

def extract_activations(model, tokenizer, prompt_list, target_layers, limit=None):
    """
    Runs inference and collects activations for ALL target layers.
    Returns a dictionary: {layer_idx: concatenated_tensor_of_activations}
    """
    # Test Mode Limit
    if limit is not None:
        print(f"TEST MODE: Processing only {limit} prompts.")
        prompt_list = prompt_list[:limit]

    # Initialise storage: {16: [], 17: [], ...}
    activations_storage = {layer: [] for layer in target_layers}
    
    # Register the multi-layer hook
    hook = MultiLayerActivationHook()
    hook.register(model, target_layers)
    
    print(f"Extracting from {len(prompt_list)} prompts across {len(target_layers)} layers...")
    
    for prompt in tqdm(prompt_list, desc="Processing prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            model(**inputs)
        
        # After forward pass, grab data from hook and clear it for next pass
        for layer in target_layers:
            act = hook.layer_activations.get(layer)
            if act is not None:
                activations_storage[layer].append(act)
        
        # Reset hook storage for the next prompt
        hook.layer_activations = {}

    hook.remove()
    
    # Stack lists into tensors
    # Result: {16: Tensor[500, 4096], 17: Tensor[500, 4096], ...}
    final_results = {}
    for layer, act_list in activations_storage.items():
        if act_list:
            final_results[layer] = torch.stack(act_list)
        else:
            print(f"WARNING: No data collected for layer {layer}")
    
    return final_results

def calculate_steering_vectors(pos_dict, neg_dict, target_layers):
    """
    Calculates steering vectors for ALL layers.
    Returns: {layer_idx: steering_vector_tensor}
    """
    print("Calculating steering vectors for all layers...")
    steering_vectors = {}
    
    for layer in target_layers:
        pos_acts = pos_dict.get(layer)
        neg_acts = neg_dict.get(layer)
        
        if pos_acts is None or neg_acts is None:
            print(f"Skipping layer {layer} (missing data).")
            continue
            
        # Mean and Subtract
        avg_pos = torch.mean(pos_acts, dim=0)
        avg_neg = torch.mean(neg_acts, dim=0)
        vector = avg_pos - avg_neg
        
        steering_vectors[layer] = vector
        
    print(f"Calculated {len(steering_vectors)} vectors.")
    return steering_vectors

# --- 3. Main Wrapper Function ---

def generate_moral_vectors(model_id, target_layers, output_filename, test_run_limit=None):
    """
    Main wrapper to generate and save a DICTIONARY of vectors.
    """
    print("--- Starting Multi-Layer Moral Vector Generation ---")
    
    # 1. Load Prompts
    pos_prompts, neg_prompts = unzip_steering_prompts()
    if not pos_prompts: return

    # 2. Load Model
    model, tokenizer = load_model_and_tokenizer(model_id)

    # 3. Extract All Layers (Positive)
    pos_dict = extract_activations(model, tokenizer, pos_prompts, target_layers, limit=test_run_limit)
    
    # 4. Extract All Layers (Negative)
    neg_dict = extract_activations(model, tokenizer, neg_prompts, target_layers, limit=test_run_limit)

    # 5. Calculate All Vectors
    vector_dict = calculate_steering_vectors(pos_dict, neg_dict, target_layers)

    # 6. Save Dictionary
    output_path = STEERING_PROMPT_DIR / output_filename
    try:
        torch.save(vector_dict, output_path)
        print(f"Saved multi-layer vector dictionary to: {output_path}")
        print(f"Contains layers: {list(vector_dict.keys())}")
    except Exception as e:
        print(f"ERROR: Could not save file. Error: {e}")

    # Cleanup
    del model, tokenizer, pos_dict, neg_dict
    torch.cuda.empty_cache()
    print("--- Generation Complete ---")