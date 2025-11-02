import pathlib

# --- Path Setup ---
ROOT_DIR = pathlib.Path(__file__).parent.parent

# --- Data Directories ---
DATA_DIR = ROOT_DIR / "data"
ORIGINAL_DATA_DIR = DATA_DIR / "original"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVAL_RESULTS_DIR = DATA_DIR / "evaluation_results"

# --- Processed Sub-directories ---
CONTRASTIVE_DATA_DIR = PROCESSED_DATA_DIR / "contrastive_datasets"
STEERING_PROMPT_DIR = PROCESSED_DATA_DIR / "steering_prompts"

# --- Original Data File ---
ORIGINAL_DATA_PATH = ORIGINAL_DATA_DIR / "SharedResponses.csv"

# --- Processed Contrastive Pair Filenames ---
U_VS_NON_U_PATH = CONTRASTIVE_DATA_DIR / "utilitarian_vs_non_utilitarian.csv"
D_VS_NON_D_PATH = CONTRASTIVE_DATA_DIR / "deontological_vs_non_deontological.csv"
U_VS_D_CONFLICT_PATH = CONTRASTIVE_DATA_DIR / "utilitarian_vs_deontological_conflict.csv"

# --- Generated Steering Prompt Filename ---
STEERING_PROMPT_PATH = STEERING_PROMPT_DIR / "aug_steer_prompts_UvD.json"

# --- Parameters ---
SAMPLE_SIZE = 500
NUM_PROMPT_PAIRS_TO_GENERATE = 500 # For prompt_generation script

# --- Model Lists ---
MODEL_LIST = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-7b-it"
]

# List of all character columns needed for parsing
CHARACTER_COLS = [
    'Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan', 'OldWoman',
    'Boy', 'Girl', 'Homeless', 'LargeWoman', 'LargeMan', 'Criminal',
    'MaleExecutive', 'FemaleExecutive', 'FemaleAthlete', 'MaleAthlete',
    'FemaleDoctor', 'MaleDoctor', 'Dog', 'Cat'
]

# --- Output Naming Templates ---
EVAL_RESULTS_CSV_TEMPLATE = EVAL_RESULTS_DIR / "{model_name}_{dataset_name}_results.csv"
EVAL_SUMMARY_TXT_TEMPLATE = EVAL_RESULTS_DIR / "{model_name}_{dataset_name}_summary_report.txt"