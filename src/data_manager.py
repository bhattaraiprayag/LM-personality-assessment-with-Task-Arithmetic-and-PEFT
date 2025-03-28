# src/data_manager.py
"""
Module for managing data loading, preprocessing,
tokenization, and DataLoader creation for language modeling tasks.
"""
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
from datasets import Dataset
from datasets import Dataset as HFDataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.utils.helper import print_output


def seed_worker(worker_id) -> None:
    """
    Sets seeds for random number generators in worker
    processes for data loading to ensure reproducibility.

    Args:
        worker_id (int): Unique identifier of the worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DataManager(pl.LightningDataModule):
    """
    LightningDataModule for preparing and loading datasets for
    language model training.

    Attributes:
        args (Namespace): Configuration arguments for data preparation
            and model training.
        data_path (str): Base path to the dataset directory.
        tokenizer (AutoTokenizer): Tokenizer used for encoding text data.
        data_collator (DataCollatorForLanguageModeling): Data collator
            for dynamic batching.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses for data loading.
        seed (int): Random seed for reproducibility.
        generator (torch.Generator): Random number generator with a fixed seed.
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

    def choose_dataset(self) -> str:
        """
        Determines the file path of the dataset based on the selected
        dataset and split.

        Returns:
            str: The path to the chosen dataset file.
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
        # elif self.args.dataset == "jigsaw":  # Jigsaw toxicity dataset
        #     base = "jigsaw_combined_for_clm.csv"
        #     path = f"{self.data_path}{self.args.dataset}/"
        #     if self.args.split is not None:
        #         if self.args.split == "base":
        #             path += base
        #         else:
        #             print(
        #                 f"Split: {self.args.split} is not supported for Jigsaw dataset. "
        #                 f"Reverting to {base}."
        #             )
        #             path += base
        elif self.args.dataset == "emotion":
            return "USE_HF_TWEET_EVAL_EMOTION"
        else:
            raise ValueError(f"Dataset '{self.args.dataset}' is not supported.")
        return path

    def load_huggingface_tweet_eval_emotion(self) -> pd.DataFrame:
        """
        Loads the 'emotion' subset of TweetEval from HuggingFace,
        filters for anger(0), joy(2), sadness(3),
        removes occurrences of '@user' or '@ user' from text,
        filters for the split passed in args (anger, joy, or sadness),
        returns the resulting data as a Pandas DataFrame with columns [text, label].
        """
        ds_dict = load_dataset("cardiffnlp/tweet_eval", "emotion")
        combined = pd.concat(
            [
                ds_dict["train"].to_pandas(),
                ds_dict["validation"].to_pandas(),
                ds_dict["test"].to_pandas(),
            ],
            ignore_index=True
        )
        anger, sadness, joy, optimism = 0, 1, 3, 2
        # combined = combined[combined["label"].isin([0, 1, 3])].copy()   # Filter for labels anger(0), joy(1), sadness(3)
        combined["text"] = combined["text"].str.replace(r"@ ?user", "", regex=True) # Remove '@user' or '@ user' from text
        combined["text"] = combined["text"].str.replace(r"&amp;", "and", regex=True)    # Replace '&amp;' with 'and'
        combined["text"] = combined["text"].str.replace(r"&lt;", "<", regex=True)   # Replace '&lt;' with '<'
        combined["text"] = combined["text"].str.replace(r"&gt;", ">", regex=True)   # Replace '&gt;' with '>'
        combined["text"] = combined["text"].str.replace(r"\n", " ", regex=True)   # Replace newline characters with spaces
        filtered = None
        if self.args.split == "anger":
            filtered = combined[combined["label"] == anger].copy()
        elif self.args.split == "joy":
            filtered = combined[combined["label"] == joy].copy()
        elif self.args.split == "sadness":
            filtered = combined[combined["label"] == sadness].copy()
        elif self.args.split == "optimism":
            filtered = combined[combined["label"] == optimism].copy()
        else:
            raise ValueError(f"Split '{self.args.split}' is not supported for the 'emotion' dataset.")
        return filtered

    def prepare_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the dataset and splits it into training, validation,
        and test sets.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames
                for training, validation, and test sets.
        """
        path = self.choose_dataset()
        if path == "USE_HF_TWEET_EVAL_EMOTION":
            dataset = self.load_huggingface_tweet_eval_emotion()
        else:
            dataset = pd.read_csv(path)
        if self.args.dataset == "pandora":
            dataset = dataset.rename(columns={"body": "text"})
        # if self.args.dataset == "jigsaw":
        #     dataset = dataset.rename(columns={"comment_text": "text"})
        print_output(f"Original Dataset: {len(dataset)} rows after loading")
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
        print_output(
            f"Train: {len(train)} / {len(train) / len(dataset) * 100:.2f}% | "
            f"Val: {len(val)} / {len(val) / len(dataset) * 100:.2f}% | "
            f"Test: {len(test)} / {len(test) / len(dataset) * 100:.2f}%"
        )
        return train, val, test

    def tokenize_dataset(self, dataset: pd.DataFrame) -> HFDataset:
        """
        Tokenizes the text data in the dataset using the tokenizer.

        Args:
            dataset (Union[pd.DataFrame, Dataset]): The dataset containing
                text data to tokenize.

        Returns:
            Dataset: A Hugging Face Dataset with tokenized inputs.
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
            dataset = HFDataset.from_pandas(dataset)
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

    def save_tokenized_dataset(self, dataset: HFDataset, trait_split: str,
                               split_type: str, subset: Optional[int] = None) -> None:
        """
        Saves the tokenized dataset to a Parquet file for faster loading
        in future runs.

        Args:
            dataset (Dataset): The tokenized dataset to save.
            trait_split (str): Identifier for the trait split (e.g.,
                'openness-top-5').
            split_type (str): Type of data split ('train', 'val',
                or 'test').
            subset (Optional[int]): Number of samples in the subset,
                if applicable.
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

    def load_tokenized_dataset(self, trait_split: str, split_type: str,
                               subset: Optional[int] = None) -> HFDataset:
        """
        Loads a previously saved tokenized dataset from a Parquet file.

        Args:
            trait_split (str): Identifier for the trait split.
            split_type (str): Type of data split.
            subset (Optional[int]): Number of samples in the subset,
                if applicable.

        Returns:
            Dataset: The loaded tokenized dataset.
        """
        subset_str = f"-{subset}" if subset else ""
        filename = f"{trait_split}-{split_type}-seed{self.seed}{subset_str}.parquet"
        load_from_path = f"{self.data_path}{self.args.dataset}/tokenized/{filename}"
        df = pd.read_parquet(load_from_path)
        dataset = HFDataset.from_pandas(df)
        return dataset

    def prepare_data(self) -> None:
        """
        Prepares the data for training. To be implemented in future.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the DataManager by preparing data splits and
        tokenizing datasets.

        Args:
            stage (Optional[str]): Stage of the setup ('fit', 'validate',
                'test', or 'predict').
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
            self.save_tokenized_dataset(self.tokenized_train,
                                        self.args.split,
                                        "train", self.args.subset)
            self.save_tokenized_dataset(self.tokenized_val,
                                        self.args.split, "val",
                                        self.args.subset)
            self.save_tokenized_dataset(self.tokenized_test,
                                        self.args.split, "test",
                                        self.args.subset)

    def train_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training data.
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

    def val_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation data.
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

    def test_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for test data.
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
