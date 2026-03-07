# Source Code Documentation

## Script Directory Structure

This directory contains scripts for vector generation, model intervention, and benchmarking.

### Configuration & Infrastructure
* **`config.py`**: Defines all file paths, model versions, and global parameters (including the `NUM_PROMPT_PAIRS = 500` standard) to ensure consistency across all notebooks and scripts.
* **`model_utils.py`**: Handles the heavy lifting of loading the LLM and Tokenizer. It manages quantization (8-bit loading) and device allocation (GPU/CPU) to prevent Out-Of-Memory errors.
* **`utils.py`**: General helper functions for file I/O, logging, and data formatting used throughout the project.

### Data Generation
* **`scenario_builder.py`**: Constructs the raw text scenarios (e.g., "A runaway trolley is heading towards..."). It populates templates with character attributes defined in the Moral Machine dataset.
* **`prompt_generation.py`**: Processes the raw CSV data into structured prompt objects. It handles the initial parsing of the Moral Machine dataset.
* **`steering_prompt_generator.py`**: Specifically formats prompts for the *Contrastive* task. It creates the "Pair" objects (Prompt A vs. Prompt B) required to calculate the difference vectors.

### Core Logic (Steering)
* **`ethical_frameworks.py`**: Defines the ground truth logic for Utilitarian vs. Deontological choices. It contains the rules that determine which action corresponds to which ethical framework.
* **`steering.py`**: Takes the contrastive prompts, runs them through the model, extracts the internal activations, and calculates the steering vector (Mean Difference) for the target layers.
* **`steering_benchmarks.py`**: Loads the generated vectors and applies them to the model during inference. It implements the **Steering Hook**, which injects the vector into the residual stream, and measures the resulting shift in behavior (the "Score" and "Invalid Rate").

### Analysis
* **`analysis.py`**: Handles analysis logic for layer by layer evaluation of total control bandwidth and refusal wall.
* **`visualisation.py`**: Contains functions to generate plots (e.g., S-Curves, Heatmaps) and visualisations of the activation spaces (PCA plots) for the final report.