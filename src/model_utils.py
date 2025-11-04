import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

def load_model_and_tokenizer(model_id):
    """
    Loads a model and its tokenizer in 4-bit mode.
    """
    print(f"\n[+] Loading model: {model_id}...")
    
    load_start_time = time.time()
    
    # Configure 4-bit quantisation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto", # Automatically uses the GPU
        trust_remote_code=True,
        trust_remote_code=True
    )
    
    load_end_time = time.time()
    print(f"[+] Model loaded successfully in {load_end_time - load_start_time:.2f} seconds.")
    
    return model, tokenizer