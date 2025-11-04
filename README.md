# LLM-Moral-Steering

> **Probing and steering LLM moral alignment using activation engineering on ethical dilemmas from the Moral Machine dataset.**

## Introduction
LLM alignment for complex, normative ethical questions (like utilitarian vs. deontological trade-offs) is a significant challenge. Traditional alignment methods (e.g., RLHF) often train for an "average" preference and can be opaque.

This project explores a more precise and interpretable method: **activation steering**. The goal is to controllably "steer" an LLM's moral reasoning at inference time, without needing to retrain the model.

## Methodology
The methodology is broken down into three stages: data processing, vector generation, and experimental steering.

## DONE

### 1. The Data (Moral Machine)
* **Source**: We use the [Moral Machine dataset](https://www.nature.com/articles/s41586-018-0637-6), a collection of 40 million human judgments on "trolley problem" style ethical dilemmas.
* **Relevance**: This dataset is ideal because it provides a massive corpus of scenarios that directly pit a utilitarian choice against a deontological one.
* **Processing**: Raw CSV data is processed into clean, text-based scenariosfor the LLM.

## IN PROGRESS

### 2. Generating the "Moral Vector" 
* **Concept**: We apply activation steering, which involves adding a "steering vector" to a model's internal activations to influence its output.
* **Generation**: We find this vector by running the model on pairs of scenarios with opposing moral frames (e.g., one strongly utilitarian, one strongly deontological). We then extract the differential activations from a specific layer and average them to create a single, robust "utilitarian-vs-deontological" steering vector

## TODO

### 3. Experiments
* **Hypothesis**: Apply the vector with a positive multiplier will make the LLM's responses more utilitarian, while a negative multiplier will make them more deontological.
* **Testing**: The vector at various strengths (multipliers) while the LLM processes new, ambiguous moral dilemmas and measure the change in its stated preference.