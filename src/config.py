import pathlib

# --- Project Structure ---
# Dynamically finds the root folder of the project
ROOT_DIR = pathlib.Path(__file__).parent.parent

# --- Data Directories ---
# Base path for data
DATA_DIR = ROOT_DIR / "data"
# Where the raw json templates/CSVs are stored
ORIGINAL_DATA_DIR = DATA_DIR / "original"
# Where cleaned/generated files go after processing
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# Where graphs, CSVs of results, and logs go
EVAL_RESULTS_DIR = DATA_DIR / "evaluation_results"

# --- Processed Sub-directories ---
# Directory for the split datasets (e.g., Util vs Non-Util)
CONTRASTIVE_DATA_DIR = PROCESSED_DATA_DIR / "contrastive_datasets"
# Directory for the JSON files used specifically for steering vector generation
STEERING_PROMPT_DIR = PROCESSED_DATA_DIR / "steering_prompts"

# --- Original Data File ---
# The source dataset from the Moral Machine experiment
ORIGINAL_DATA_PATH = ORIGINAL_DATA_DIR / "SharedResponses.csv"

# --- Processed Contrastive Pair Filenames ---
# Paths to the specific CSV splits used to train different vector types
U_VS_NON_U_PATH = CONTRASTIVE_DATA_DIR / "utilitarian_vs_non_utilitarian.csv"
D_VS_NON_D_PATH = CONTRASTIVE_DATA_DIR / "deontological_vs_non_deontological.csv"
# The primary dataset for the U vs D conflict (The moral questions)
U_VS_D_CONFLICT_PATH = CONTRASTIVE_DATA_DIR / "utilitarian_vs_deontological_conflict.csv"

# --- Generated Steering Prompt Filename ---
# The JSON file containing the A/B pairs used to build the steering vectors
STEERING_PROMPT_PATH = STEERING_PROMPT_DIR / "steer_prompts_UvD.json"

# --- Parameters ---
# The limit for how many pairs to use for testing/benchmarking
SAMPLE_SIZE = 500
# The limit for how many pairs to use for BUILDING the vector
# 500 is the "High Definition" setting (High Signal/Noise ratio).
NUM_PROMPT_PAIRS_TO_GENERATE = 500 # For prompt_generation script

# --- Model Lists ---
# The specific versions of models targeted for this research
MODEL_LIST = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-7b-it"
]

# List of all character columns needed for parsing the Moral Machine CSV
CHARACTER_COLS = [
    'Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan', 'OldWoman',
    'Boy', 'Girl', 'Homeless', 'LargeWoman', 'LargeMan', 'Criminal',
    'MaleExecutive', 'FemaleExecutive', 'FemaleAthlete', 'MaleAthlete',
    'FemaleDoctor', 'MaleDoctor', 'Dog', 'Cat'
]

# --- Output Naming Templates ---
# Standardised naming for output files to ensure consistency across runs
EVAL_RESULTS_CSV_TEMPLATE = EVAL_RESULTS_DIR / "{model_name}_{dataset_name}_results.csv"
EVAL_SUMMARY_TXT_TEMPLATE = EVAL_RESULTS_DIR / "{model_name}_{dataset_name}_summary_report.txt"