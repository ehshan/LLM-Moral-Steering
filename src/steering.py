import torch
import json
import os
import time
from tqdm.auto import tqdm # For output progress bar

# Import from our own project files
from src.config import STEERING_PROMPT_PATH, STEERING_PROMPT_DIR, MODEL_LIST
from src.model_utils import load_model_and_tokenizer

# --- 1. Activation Hook Class ---

class ActivationHook:
    """
    A simple hook class to store the activations of a specific layer.
    This works by "catching" the output of a module during the forward pass.
    """
    def __init__(self):
        self.activation = None
        self.handle = None

    def hook_fn(self, module, input_tensors, output_tensors):
        """
        This is the function that gets called by PyTorch during the forward pass.
        """
        # The output of these layers is a TUPLE.
        # The first element, output_tensors[0], is the hidden state.
        hidden_state = output_tensors[0]
        
        # --- ISSUE LOG - TOO MANY INDICES---
        # check the dimension of the hidden_state tensor.
        
        if hidden_state.dim() == 3:
            # EXPECTED CASE: [batch_size, seq_len, hidden_dim]
            # Get the activation of the last token in the sequence.
            self.activation = hidden_state[0, -1, :].detach().cpu()
            
        elif hidden_state.dim() == 2:
            # UNEXPECTED CASE: The error implies this is happening.
            # Assume [seq_len, hidden_dim]. Get the last token.
            print(f"WARNING: Layer output was 2D (shape {hidden_state.shape}), not 3D. Taking last token.")
            self.activation = hidden_state[-1, :].detach().cpu()
            
        else:
            # UNKNOWN CASE
            print(f"ERROR: Unexpected activation shape: {hidden_state.shape}")
            self.activation = None

    def register(self, model, layer_index):
        """
        Registers this hook on a specific layer of the model.
        """
        try:
            target_layer = model.model.layers[layer_index]
            self.handle = target_layer.register_forward_hook(self.hook_fn)
            print(f"[+] Hook registered on layer {layer_index}.")
        except Exception as e:
            print(f"ERROR: Failed to register hook on layer {layer_index}. Error: {e}")
            print("Check if the model architecture (model.model.layers) is correct.")


    def remove(self):
        """
        Removes the hook from the model.
        """
        if self.handle:
            self.handle.remove()
            print(f"[+] Hook removed.")


# --- 2. Core Logic Functions ---

def unzip_steering_prompts():
    """
    Loads the JSON file and "unzips" it into two lists.
    """
    print(f"Loading steering prompts from: {STEERING_PROMPT_PATH}")
    try:
        with open(STEERING_PROMPT_PATH, 'r', encoding='utf-8') as f:
            prompts_json = json.load(f)
        
        positive_prompts = [item['DeontologicalPrompt'] for item in prompts_json]
        negative_prompts = [item['UtilitarianPrompt'] for item in prompts_json]
        
        print(f"Successfully loaded {len(positive_prompts)} positive (Deon) and {len(negative_prompts)} negative (Util) prompts.")
        return positive_prompts, negative_prompts
        
    except FileNotFoundError:
        print(f"ERROR: Steering prompts file not found at {STEERING_PROMPT_PATH}")
        return None, None
    except Exception as e:
        print(f"ERROR: Could not read JSON file. Error: {e}")
        return None, None

def extract_activations(model, tokenizer, prompt_list, target_layer_index):
    """
    Runs inference on all prompts in a list and collects the activations
    from the target layer using a hook.
    """
    activations_list = []
    hook = ActivationHook()
    hook.register(model, target_layer_index)
    
    print(f"Extracting activations from {len(prompt_list)} prompts...")
    
    # Use tqdm for a progress bar
    for prompt in tqdm(prompt_list, desc="Processing prompts"):
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Run a forward pass. We don't need to .generate() text,
        # we just need the internal activations.
        with torch.no_grad():
            model(**inputs)
        
        # The hook has now run and self.activation has been set
        if hook.activation is not None:
            activations_list.append(hook.activation)
        
    hook.remove() # Clean up the hook
    
    if not activations_list:
        print("ERROR: No activations were collected. Check hook logic.")
        return None
        
    # Stack all activations into a single tensor
    return torch.stack(activations_list)

def calculate_steering_vector(positive_activations, negative_activations):
    """
    Calculates the steering vector by averaging and subtracting.
    """
    print("Calculating steering vector...")
    
    # Calculate the mean activation for each set
    # Shape of activations: [num_prompts, hidden_dim]
    avg_positive_activations = torch.mean(positive_activations, dim=0)
    avg_negative_activations = torch.mean(negative_activations, dim=0)
    
    # Subtract to get the steering vector
    steering_vector = avg_positive_activations - avg_negative_activations
    
    print(f"Steering vector calculated. Shape: {steering_vector.shape}")
    return steering_vector

# --- 3. Main Wrapper Function ---

def generate_moral_vector(model_id, target_layer_index, output_filename):
    """
    Main function to generate and save the moral steering vector.
    """
    print("--- Starting Moral Vector Generation ---")
    
    # 1. Load Prompts
    positive_prompts, negative_prompts = unzip_steering_prompts()
    if positive_prompts is None:
        return

    # 2. Load Model
    # We import this from our new utility script!
    model, tokenizer = load_model_and_tokenizer(model_id)

    # 3. Extract Activations (Positive)
    positive_activations = extract_activations(
        model, tokenizer, positive_prompts, target_layer_index
    )
    if positive_activations is None: return

    # 4. Extract Activations (Negative)
    negative_activations = extract_activations(
        model, tokenizer, negative_prompts, target_layer_index
    )
    if negative_activations is None: return

    # 5. Calculate Vector
    steering_vector = calculate_steering_vector(
        positive_activations, negative_activations
    )

    # 6. Save the Vector
    # We save it in the same directory as the prompts
    output_path = STEERING_PROMPT_DIR / output_filename
    try:
        torch.save(steering_vector, output_path)
        print(f"Steering vector saved successfully to: {output_path}")
    except Exception as e:
        print(f"ERROR: Could not save steering vector. Error: {e}")
        
    # 7. Clean up memory
    del model, tokenizer, positive_activations, negative_activations
    torch.cuda.empty_cache()
    print("[+] Model unloaded and VRAM cleared.")
    print("--- Moral Vector Generation Complete ---")

# --- This allows the script to be run directly ---
if __name__ == "__main__":
    
    # Add project root to path to allow `from src...` imports
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # --- Configuration for direct run ---
    # We use the first model from our config list as the default
    MODEL_TO_USE = MODEL_LIST[0] 
    # This is a good default layer for an 8B model (which have ~32 layers)
    TARGET_LAYER = 20 
    OUTPUT_FILE = "deon_vs_util_vector.pt"
    
    print(f"Running script directly (locally)...")
    print(f"Model: {MODEL_TO_USE}")
    print(f"Layer: {TARGET_LAYER}")
    
    generate_moral_vector(
        model_id=MODEL_TO_USE,
        target_layer_index=TARGET_LAYER,
        output_filename=OUTPUT_FILE
    )