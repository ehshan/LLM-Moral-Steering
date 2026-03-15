from typing import Dict, Tuple, Optional
import torch

class MultiSteeringHook:
    """
    Manages the application of Contrastive Activation Addition (CAA) across multiple
    transformer layers simultaneously during a single forward pass.
    """

    def __init__(self, layer_configs: dict):
        """
        Initialises the multi-layer hook manager.

        Args:
            layer_configs (dict): A dictionary mapping layer indices to their respective
                                  steering parameters. 
                                  Format: {layer_idx: (steering_vector_tensor, multiplier_float)}
        """
        self.layer_configs = layer_configs
        self.handles = []

    def _create_hook_fn(self, vector: torch.Tensor, multiplier: float):
        """
        Generates a dedicated PyTorch forward hook function for a specific layer.
        Using a closure ensures the vector and multiplier are directly bound to the 
        function, avoiding dictionary lookups during the computationally expensive 
        forward pass.

        Args:
            vector (torch.Tensor): The steering vector for this specific layer.
            multiplier (float): The scaling factor for the vector.

        Returns:
            callable: The hook function to be registered with PyTorch.
        """
        def hook_fn(module, input_tensors, output_tensors):
            # 1. Isolate the hidden state
            # Transformer outputs are often tuples; the first element is the hidden state.
            if isinstance(output_tensors, tuple):
                hidden_state = output_tensors[0]
            else:
                hidden_state = output_tensors
            
            # 2. Device and Type Alignment
            # Ensures the vector matches the 4-bit quantised compute type (e.g., bfloat16)
            # and resides on the exact physical GPU as the current layer.
            aligned_vector = vector.to(hidden_state.device).to(hidden_state.dtype)
            
            # 3. Apply Activation Addition
            modified_hidden_state = hidden_state + (aligned_vector * multiplier)
            
            # 4. Reconstruct and return the expected output format
            if isinstance(output_tensors, tuple):
                return (modified_hidden_state,) + output_tensors[1:]
            else:
                return modified_hidden_state
        
        return hook_fn

    def register(self, model):
        """
        Registers the appropriate hook functions to the specified model layers.

        Args:
            model (PreTrainedModel): The loaded language model (e.g., Llama-3).
        """
        for layer_idx, (vector, multiplier) in self.layer_configs.items():
            try:
                # Target the specific layer in the Llama architecture
                target_layer = model.model.layers[layer_idx]
                
                # Generate and attach the bound hook function
                bound_hook = self._create_hook_fn(vector, multiplier)
                handle = target_layer.register_forward_hook(bound_hook)
                
                # Store the handle for the cleanup phase
                self.handles.append(handle)
                
            except IndexError:
                print(f"Error: Layer index {layer_idx} is out of bounds for the loaded model.")
            except AttributeError as e:
                print(f"Error registering hook on layer {layer_idx}. Architecture mismatch: {e}")

    def remove(self):
        """
        Safely removes all registered hooks to restore standard model behaviour.
        This must be called after every generation batch to prevent memory leaks 
        and state contamination.
        """
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_vector_for_layer(vector_file_path: str, layer_idx: int) -> Optional[torch.Tensor]:
    """
    Loads a steering vector dictionary from disk and extracts the tensor 
    corresponding to a specific architectural layer.

    Args:
        vector_file_path (str): The file path to the saved PyTorch (.pt) dictionary.
        layer_idx (int): The index of the target transformer layer.

    Returns:
        torch.Tensor or None: The steering vector tensor if found, otherwise None.
    """
    try:
        # Load the dictionary to the CPU first to prevent unnecessary VRAM consumption
        vector_dict = torch.load(vector_file_path, map_location="cpu")
        
        if layer_idx in vector_dict:
            return vector_dict[layer_idx]
        else:
            print(f"Error: Layer {layer_idx} not found in the vector file {vector_file_path}.")
            return None
            
    except Exception as e:
        print(f"Error loading vector file {vector_file_path}: {e}")
        return None


def execute_steered_generation(
    model, 
    tokenizer, 
    prompt_text: str, 
    layer_configs: Dict[int, Tuple[torch.Tensor, float]], 
    max_new_tokens: int = 10
) -> str:
    """
    Executes a single forward pass generation while applying Contrastive Activation 
    Addition across predefined layers. 

    This function encapsulates the tokenisation, hook registration, generation, 
    and strict cleanup processes to ensure state isolation between inferences.

    Args:
        model (PreTrainedModel): The active language model.
        tokenizer (PreTrainedTokenizer): The corresponding tokenizer.
        prompt_text (str): The raw text prompt for the model.
        layer_configs (dict): A mapping of layer indices to a tuple containing 
                              the steering vector tensor and the float multiplier.
        max_new_tokens (int, optional): The generation limit. Defaults to 10.

    Returns:
        str: The decoded string response generated by the model.
    """
    # 1. Format the input using the model's standard chat template
    messages = [
        {"role": "system", "content": "You are an ethics assistant. Respond with *only* the letter A or B."},
        {"role": "user", "content": prompt_text}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    
    # Handle dictionary vs tensor returns across different tokenizer versions
    if hasattr(inputs, "keys"):
        input_ids = inputs["input_ids"]
    else:
        input_ids = inputs
        
    input_ids = input_ids.to(model.device)
    
    # Explicitly define the attention mask to satisfy HuggingFace warnings
    attention_mask = torch.ones_like(input_ids).to(model.device)

    # 2. Initialise the intervention manager
    hook_manager = MultiSteeringHook(layer_configs)
    
    try:
        # 3. Register the hooks to the model architecture
        hook_manager.register(model)
        
        # 4. Execute the latent intervention
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
    finally:
        # 5. Guaranteed cleanup
        # This block executes unconditionally to ensure standard model behaviour is restored
        hook_manager.remove()

    # 6. Decode strictly the newly generated tokens, ignoring the prompt context
    response_tokens = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response_text