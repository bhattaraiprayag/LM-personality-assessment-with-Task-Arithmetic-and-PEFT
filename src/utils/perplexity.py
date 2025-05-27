# src/utils/perplexity.py
"""
Module for benchmarking perplexity of a language model.
"""
import argparse
import random
import time
import numpy as np
import torch

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from pytorch_lightning.utilities import rank_zero_only


BLOCK_SIZE = 512
BATCH_SIZE = 4

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

@rank_zero_only
def run_perplexity_benchmark(model, tokenizer, seed=42):
    """
    Run perplexity benchmark on WikiText-103 and Penn Treebank datasets.
    
    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer for the model
        seed: Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing perplexity scores for each dataset
    """
    class Args:
        def __init__(self):
            self.seed = seed
    
    args = Args()
    return benchmark_perplexity(model, tokenizer, args)

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
