# Interactive Experiments

This directory contains the Jupyter Notebooks used to run the experimental pipeline step-by-step.

## Notebook Overview

### 1. Data Analysis & Preparation
* **`moral_machine_analysis.ipynb`**: Exploratory Data Analysis (EDA) of the Moral Machine dataset. It visualises the distribution of ethical scenarios to ensure our inputs are balanced.
* **`moral_machine_contrastive.ipynb`**: Filters the raw data to identify scenarios that are suitable for contrastive pairs (e.g., finding "save 5 vs save 1" scenarios).
* **`mm_steering_prompt_generation.ipynb`**: The final preparation step. It takes the processed data and formats it into the specific JSON structure required for the steering vector generator.

### 2. Core Steering Pipeline
* **`vector_generation.ipynb`**: **The Generator.** 1. Loads the model (Llama-3-8B).
    2. Runs the contrastive prompt pairs through the network.
    3. Extracts hidden states and calculates the **Steering Vector** (Mean Difference) for layers 0–32.
    4. Saves the vectors as `.pt` files.

* **`steering_benchmark.ipynb`**: **The Tester.** * Loads the generated vectors.
    * Performs the **Layer Sweep**.
    * Performs the **Strength Sweep** (testing multipliers from -2.5 to +2.5).
    * Generates the final S-Curve and Heatmap visualisations.

### 3. Advanced / Experimental
* **`moral_machine_preference learning.ipynb`**: An experimental notebook exploring standard preference learning techniques (like direct preference optimisation) to compare against our activation steering results.

## Hardware Requirements
* **GPU**: These notebooks require a GPU with at least 16GB VRAM (A100 recommended, or RTX 3090/4090/5080).
* **RAM**: 32GB system RAM recommended for loading the 8B model.