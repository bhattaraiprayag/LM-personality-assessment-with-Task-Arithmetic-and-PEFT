# src/utils/perplexity.py
"""
Module for benchmarking perplexity of a language model.
"""

import argparse
import random
import time
import os
import json
import csv
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from peft.helpers import rescale_adapter_scale
from peft.tuners.lora import LoraLayer

BLOCK_SIZE = 1024
BATCH_SIZE = 8

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize_function(examples, tokenizer, text_column):
    return tokenizer(examples[text_column], return_special_tokens_mask=True)

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [concatenated[k][i: i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k in concatenated
    }
    return result

def benchmark_perplexity(model, tokenizer, args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = {}

    for dataset_key, dataset_info in {
        "wikitext103": {
            "name": "WikiText-103",
            "load_args": ("wikitext", "wikitext-103-v1"),
            "text_column": "text"
        },
        "penntree": {
            "name": "Penn Treebank",
            "load_args": ("ptb_text_only", "penn_treebank"),
            "text_column": "sentence",
            "extra_kwargs": {"trust_remote_code": True}
        }
    }.items():
        # print(f"\nLoading {dataset_info['name']} splits...")
        kwargs = dataset_info.get("extra_kwargs", {})
        test_dataset = load_dataset(*dataset_info["load_args"], split="test", **kwargs)
        val_dataset = load_dataset(*dataset_info["load_args"], split="validation", **kwargs)
        dataset = concatenate_datasets([test_dataset, val_dataset])

        # print(f"Samples in {dataset_info['name']}: {len(dataset)}")

        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer, dataset_info["text_column"]),
            batched=True,
            remove_columns=[dataset_info["text_column"]]
        )

        lm_dataset = tokenized_dataset.map(group_texts, batched=True)

        dataloader = DataLoader(
            lm_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )

        total_loss, total_tokens = 0.0, 0
        # print(f"Evaluating on {dataset_info['name']}...")

        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_info['name']}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                num_tokens = attention_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        elapsed = time.time() - start_time

        # print(f"Evaluation completed in {elapsed:.2f} seconds.")
        # print(f"Perplexity on {dataset_info['name']}: {perplexity:.2f}")
        results[dataset_key] = perplexity

    return results

def benchmark_model_perplexity(model, tokenizer, seed=42, is_lora=False, lora_scales=None):
    """
    Runs perplexity benchmarks on the model, with support for LoRA models and multiple scales.
    
    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer for the model
        seed: Random seed for reproducibility
        is_lora: Boolean indicating if the model is a LoRA model
        lora_scales: List of scales to evaluate for LoRA models (if is_lora=True)
        
    Returns:
        dict: Dictionary containing perplexity results. For vanilla models, this is a simple
              dictionary with dataset names as keys. For LoRA models, it's a nested dictionary
              with scales as the first level keys and datasets as second level keys.
    """
    args = argparse.Namespace()
    args.seed = seed
    
    if not is_lora:
        # For vanilla models, just run the benchmark once
        return benchmark_perplexity(model, tokenizer, args)
    else:
        # For LoRA models, run benchmark at each scale
        results = {}
        for scale in lora_scales:
            # Check if the model contains LoRA layers
            if any(isinstance(module, LoraLayer) for module in model.modules()):
                with rescale_adapter_scale(model, scale):
                    scale_results = benchmark_perplexity(model, tokenizer, args)
                results[str(scale)] = scale_results
            else:
                # If no LoRA layers found, run as vanilla model
                results[str(scale)] = benchmark_perplexity(model, tokenizer, args)
        return results

def save_perplexity_results(output_dir, experiment_id, results, is_lora=False):
    """
    Saves perplexity benchmark results to the appropriate files.
    
    Args:
        output_dir: Base output directory
        experiment_id: Unique experiment ID
        results: Perplexity results dict
        is_lora: Boolean indicating if this is for a LoRA model
    """
    exp_dir = os.path.join(output_dir, experiment_id)
    
    if not is_lora:
        # For vanilla models, save directly to the metadata
        # We'll handle metadata update in the main script
        return results
    else:
        # For LoRA models, save to a separate CSV file
        csv_file = os.path.join(exp_dir, "perplexity_benchmarks.csv")
        
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['scale', 'dataset', 'perplexity']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for scale, scale_results in results.items():
                for dataset, perplexity in scale_results.items():
                    writer.writerow({
                        'scale': scale,
                        'dataset': dataset,
                        'perplexity': perplexity
                    })
        
        # Return path to CSV file
        return os.path.join(experiment_id, "perplexity_benchmarks.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed value")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    results = benchmark_perplexity(model, tokenizer, args)
    print("\nFinal Perplexity Results (JSON):")
    print(results)
