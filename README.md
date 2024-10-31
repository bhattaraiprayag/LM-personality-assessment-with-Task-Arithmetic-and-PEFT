# Table of Contents
1. [Introduction](#introduction)
2. [Objective](#objective)
3. [Installation](#installation)
4. [Dataset Description](#dataset-description)
5. [Experimental Design](#experimental-design)
6. [Model Architecture](#model-architecture)
7. [Implementation](#implementation)
8. [Data Preprocessing](#data-preprocessing)
9. [Data Management](#data-management)
10. [Model Training](#model-training)
11. [Evaluation](#evaluation)
12. [How to Run the Code](#how-to-run-the-code)
    - [Required Arguments](#required-arguments)
    - [Example Commands](#example-commands)
13. [Results](#results)
14. [Contributions](#contributions)
15. [References](#references)

## Introduction
This codebase accompanies a master's thesis that investigates the personality analysis of large language models (LLMs) using adapter-based techniques like Low-Rank Adaptation (LoRA) and Task Arithmetic. The study aims to understand how fine-tuning LLMs, specifically GPT-2, on personality-specific datasets affects their behavior, evaluated through personality assessments based on both OCEAN (Big Five) and MBTI traits.

## Objective
The primary objective of this thesis is to explore how personality traits in a GPT-2 model can be influenced using techniques such as Task Arithmetic and adapter-based methods (LoRA). The goal is to fine-tune GPT-2 on personality-specific splits derived from the Pandora dataset and analyze the resulting changes in personality traits.

## Installation
To set up the environment for this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/username/repo.git
    cd repo
    ```

2. Set up a Python virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4. Additional dependencies (if required):
    - For NLP tasks: `transformers`, `datasets`
    - For model training: `torch`, `pytorch-lightning`

## Dataset Description
The Pandora dataset is used for this project. It includes Reddit comments linked to specific personality traits from both OCEAN (Big Five) and MBTI personality dimensions.

- **Comments Dataset (all_comments.csv)**: Contains user comments.
- **Author Profiles (author_profiles.csv)**: Contains personality trait labels for authors.

**Dataset Statistics**:
- Total unique authors: 10,296
- Authors with valid OCEAN traits: 1,605-1,608 (depending on trait)
- Authors with valid MBTI traits: 9,074-9,083

**Personality Trait Information**:
For each trait (OCEAN and MBTI), multiple subsets are created by ranking authors and selecting the top/bottom k% based on their scores to simulate "extreme" personality conditions.

## Experimental Design
The key focus is to simulate Task Arithmetic on personality dimensions by fine-tuning GPT-2 on personality-specific data splits. The process includes:

1. **Dataset Splits**:
    - Dividing the dataset into splits for each personality dimension (both OCEAN and MBTI) by selecting top/bottom k% authors.
    - Creating "extreme" personality datasets for model fine-tuning.

2. **Fine-Tuning**:
    - Applying LoRA adapters to the GPT-2 model to inject task-specific information.
    - Scaling adapter weights (ranging from 0.1 to 1.0, with step increments of 0.1) to observe personality variation.

3. **Evaluation**:
    - Evaluating the model's personality using predefined personality inventories to assess model behavior before and after fine-tuning.

## Model Architecture
The model used for this project is GPT-2, a causal language model. Key components include:

- **LoRA (Low-Rank Adaptation)**: Adapter-based fine-tuning for efficient parameter updates.
- **Task Arithmetic**: Arithmetic operations on personality traits to understand their impact on model behavior.

## Implementation
### Data Preprocessing
The data preprocessing steps involve:
- Loading and cleaning Reddit comments.
- Filtering comments based on personality data.
- Creating subsets for specific personality splits.

**Relevant File**: `src/data_preprocessor.py`

### Data Management
Data management steps include:
- Splitting the data into training, validation, and test sets.
- Tokenizing the data using HuggingFace transformers.
- Saving tokenized data in an efficient format.

**Relevant File**: `src/data_manager.py`

### Model Training
The model is trained using the PyTorch Lightning framework:
- Loading the pre-trained GPT-2 from HuggingFace.
- Adding LoRA adapters to enhance personality adaptation.
- Using the causal language modeling (CLM) objective for fine-tuning.

**Relevant File**: `src/model_manager.py`

### Evaluation
Evaluation is conducted using custom methods that align the model's responses with personality traits.
- **Relevant File**: `src/eval_manager.py`

## How to Run the Code
### Required Arguments
- `--data_path`: Path to the dataset directory | ==> options: 'pandora', 'toxicity', ...
- `--model_name`: Pre-trained model to use | ==> options: `gpt2`, `gpt3`, ...
- `--batch_size`: Batch size for training | ==> required
- `--epochs`: Number of epochs for training | ==> required
- `--subset`: Optional, subset size for prototyping.

### Example Commands
**1. Baseline Fine-Tuning without PEFT (Full Dataset)**
```bash
python start_experiment.py --data_path="data/" --model_name="gpt2" --batch_size=16 --epochs=3
```

**2. Baseline Fine-Tuning without PEFT (Subset for Prototyping)**
```bash
python start_experiment.py --data_path="data/" --model_name="gpt2" --batch_size=16 --epochs=3 --subset=5000
```

**3. Fine-Tuning with PEFT (LoRA Adapter, Full Dataset)**
```bash
python start_experiment.py --data_path="data/" --model_name="gpt2" --batch_size=16 --epochs=3 --use_peft="lora" --scale_peft=0.5
```

**4. Fine-Tuning with PEFT (LoRA Adapter, Subset for Prototyping)**
```bash
python start_experiment.py --data_path="data/" --model_name="gpt2" --batch_size=16 --epochs=3 --use_peft="lora" --scale_peft=0.5 --subset=5000
```

## Results
The results of the experiments include:
- Training and validation losses.
- Personality evaluation metrics pre- and post-fine-tuning.
- Visualizations of the impact of scaling LoRA adapters on model behavior.

**Results Location**: Outputs are saved in the `outputs/` directory, including raw metrics and visualized graphs.

## Contributions
This project is part of a master's thesis by Prayag Bhattarai. The contributions include:
- Conceptualizing and designing the experimental framework.
- Implementing data preprocessing, model fine-tuning, and evaluation procedures.
- Analyzing the experimental results.

## References
- [Huggingface Transformers](https://huggingface.co/transformers/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- Siebels, F. (2024). Master Thesis on Emotion Detection with NLP.
- Ruder, S. et al. (2022). Modular Deep Learning Tutorial. EMNLP 2022.

