# src/data_manager.py

"""
Data management module for handling datasets and data loaders.
"""

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


def seed_worker(worker_id):
    """
    Initializes a unique random seed for each worker process in DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DataManager(pl.LightningDataModule):
    """
    Manages data loading, preprocessing, and splitting for the experiment.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_path = "data/"
        self.tokenized_train = None
        self.tokenized_val = None
        self.tokenized_test = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        self.batch_size = self.args.batch_size
        self.num_workers = self.args.num_workers
        self.seed = self.args.seed
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def choose_dataset(self):
        """
        Choose the dataset based on the dataset name and split.
        """
        if self.args.dataset == "pandora":
            base = "all_comments_since_2015_chunk_0.csv"
            path = f"{self.data_path}{self.args.dataset}/"
            if self.args.split is not None:
                if self.args.split == "base":
                    path += base
                else:
                    # Example split: "openness-top-5"
                    path += f"splits_balanced/{self.args.split}.csv"
            else:
                path += base
        elif self.args.dataset == "jigsaw":  # Jigsaw toxicity dataset
            base = "jigsaw_combined_for_clm.csv"
            path = f"{self.data_path}{self.args.dataset}/"
            if self.args.split is not None:
                if self.args.split == "base":
                    path += base
                else:
                    print(
                        f"Split: {self.args.split} is not supported for Jigsaw dataset. "
                        f"Reverting to {base}."
                    )
                    path += base
        else:
            raise ValueError(f"Dataset '{self.args.dataset}' is not supported.")
        return path

    def prepare_splits(self):
        """
        Prepare the train, validation, and test splits.
        """
        path = self.choose_dataset()
        dataset = pd.read_csv(path)
        print(f"Original Dataset: {len(dataset)}")
        if self.args.dataset == "pandora":
            dataset = dataset.rename(columns={"body": "text"})
        if self.args.dataset == "jigsaw":
            dataset = dataset.rename(columns={"comment_text": "text"})
        train_val, test = train_test_split(
            dataset, test_size=0.05, random_state=self.args.seed
        )
        train, val = train_test_split(
            train_val, test_size=float(0.05 / 0.95), random_state=self.args.seed
        )
        val_subset = max(1, int(self.args.subset * 0.1)) if self.args.subset else None
        test_subset = max(1, int(self.args.subset * 0.1)) if self.args.subset else None
        train = (
            train.sample(n=self.args.subset, random_state=self.args.seed)
            if self.args.subset
            else train
        )
        val = (
            val.sample(n=val_subset, random_state=self.args.seed) if val_subset else val
        )
        test = (
            test.sample(n=test_subset, random_state=self.args.seed)
            if test_subset
            else test
        )
        print(
            f"Train: {len(train)} / {len(train) / len(dataset) * 100:.2f}% | "
            f"Val: {len(val)} / {len(val) / len(dataset) * 100:.2f}% | "
            f"Test: {len(test)} / {len(test) / len(dataset) * 100:.2f}%"
        )
        return train, val, test

    def tokenize_dataset(self, dataset: pd.DataFrame) -> Dataset:
        """
        Tokenize the dataset using the tokenizer.
        """

        def tokenize_seqs(examples: Dict[str, List[str]]) -> Dict[str, Any]:
            texts_with_special_tokens = [
                self.tokenizer.bos_token + text + self.tokenizer.eos_token
                for text in examples["text"]
            ]
            tokenized_output = self.tokenizer(
                texts_with_special_tokens,
                truncation=True,
                max_length=768,
                padding=False,
            )
            tokenized_output["labels"] = tokenized_output["input_ids"].copy()
            return tokenized_output

        if isinstance(dataset, pd.DataFrame):
            dataset = Dataset.from_pandas(dataset)
        tokenized_dataset = dataset.map(tokenize_seqs, batched=True)
        columns_to_keep = ["input_ids", "attention_mask"]
        tokenized_dataset = tokenized_dataset.remove_columns(
            [
                col
                for col in tokenized_dataset.column_names
                if col not in columns_to_keep
            ]
        )
        return tokenized_dataset

    def save_tokenized_dataset(
        self,
        dataset: Dataset,
        trait_split: str,
        split_type: str,
        subset: Optional[int] = None,
    ):
        """
        Save the tokenized dataset as a Parquet file.
        """
        df = dataset.to_pandas()
        subset_str = f"-{subset}" if subset else ""
        filename = f"{trait_split}-{split_type}-seed{self.seed}{subset_str}.parquet"
        table = pa.Table.from_pandas(df)
        save_to_path = f"{self.data_path}{self.args.dataset}/tokenized/{filename}"
        os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
        pq.write_table(
            table, save_to_path, compression="zstd", use_dictionary=True, version="2.6"
        )

    def load_tokenized_dataset(
        self, trait_split: str, split_type: str, subset: Optional[int] = None
    ):
        """
        Load the tokenized dataset from a Parquet file.
        """
        subset_str = f"-{subset}" if subset else ""
        filename = f"{trait_split}-{split_type}-seed{self.seed}{subset_str}.parquet"
        load_from_path = f"{self.data_path}{self.args.dataset}/tokenized/{filename}"
        df = pd.read_parquet(load_from_path)
        dataset = Dataset.from_pandas(df)
        return dataset

    def prepare_data(self):
        """
        Placeholder method for data preparation.
        """
        # # TO DO: To be implemented later.

    def setup(self, stage: Optional[str] = None):
        """
        Setup the data for different stages (fit, test).
        """
        train_df, val_df, test_df = self.prepare_splits()
        try:
            self.tokenized_train = self.load_tokenized_dataset(
                self.args.split, "train", self.args.subset
            )
            self.tokenized_val = self.load_tokenized_dataset(
                self.args.split, "val", self.args.subset
            )
            self.tokenized_test = self.load_tokenized_dataset(
                self.args.split, "test", self.args.subset
            )
        except FileNotFoundError:
            self.tokenized_train = self.tokenize_dataset(train_df)
            self.tokenized_val = self.tokenize_dataset(val_df)
            self.tokenized_test = self.tokenize_dataset(test_df)
            self.save_tokenized_dataset(
                self.tokenized_train, self.args.split, "train", self.args.subset
            )
            self.save_tokenized_dataset(
                self.tokenized_val, self.args.split, "val", self.args.subset
            )
            self.save_tokenized_dataset(
                self.tokenized_test, self.args.split, "test", self.args.subset
            )

    def train_dataloader(self):
        """
        Create the train dataloader.
        """
        return DataLoader(
            self.tokenized_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    def val_dataloader(self):
        """
        Create the validation dataloader.
        """
        return DataLoader(
            self.tokenized_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    def test_dataloader(self):
        """
        Create the test dataloader.
        """
        return DataLoader(
            self.tokenized_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )
