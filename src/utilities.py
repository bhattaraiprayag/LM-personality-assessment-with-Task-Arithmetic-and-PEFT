# src/utilities.py

"""
Utility functions and classes for the personality assessment project.
"""

import hashlib
import json
import logging
import math
import os
import random
import warnings
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from dotenv import load_dotenv
from lightning.pytorch import seed_everything
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities import rank_zero_only
from transformers import HfArgumentParser
from transformers import set_seed as hf_set_seed

from src.data_manager import DataManager
from src.model_manager import CLMModel


@dataclass
class ExperimentArguments:
    """
    Arguments needed to run task-arithmetic experiments with PEFT-based LLM fine-tuning for
    personality assessment.
    """

    dataset: str = field(
        default="pandora",
        metadata={"help": "| Dataset to use | ==> options: 'pandora', 'jigsaw', ..."},
    )
    split: Optional[str] = field(
        default="base",
        metadata={
            "help": (
                "| Dataset split to use. Based on top/bottom k-percentile authors comments "
                "for that trait label | ==> options: 'base', 'conscientiousness-top-5', "
                "'openness-bot-5', ..."
            )
        },
    )
    subset: Optional[int] = field(
        default=None,
        metadata={
            "help": "| Number of samples to subset (to ease prototyping) | ==> optional"
        },
    )
    output: str = field(
        default="outputs/",
        metadata={"help": "| Directory to experiment outputs | ==> required"},
    )
    model_name: str = field(
        default="gpt2",
        metadata={
            "help": "| Pre-trained model to use | ==> options: 'gpt2', 'gpt3', ..."
        },
    )
    seed: int = field(
        default=183, metadata={"help": "| Seed for reproducibility | ==> required"}
    )
    epochs: int = field(
        default=1, metadata={"help": "| Number of epochs for training | ==> required"}
    )
    batch_size: int = field(
        default=16, metadata={"help": "| Batch size for training | ==> required"}
    )
    lr: float = field(
        default=1e-5, metadata={"help": "| Learning rate for training | ==> required"}
    )
    grad_steps: int = field(
        default=4,
        metadata={
            "help": "| Number of gradient steps to accumulate before backprop | ==> required"
        },
    )
    use_peft: Optional[str] = field(
        default=None,
        metadata={
            "help": "| PEFT method to use | ==> options: 'lora', 'prompt-tuning', ..."
        },
    )
    scale_peft: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "| Scaling factor for PEFT weights. If set to 1, PEFT module's weights are "
                "not being scaled. | ==> optional"
            )
        },
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={
            "help": (
                "Ratio of warmup steps for the scheduler. Acceptable range: from 0.03 to "
                "0.1 | optional"
            )
        },
    )
    num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of CPU workers for DataLoader | optional"},
    )
    accelerator: Optional[str] = None
    devices: Optional[List[int]] = None


class Utilities:
    """
    A collection of static utility methods for experiment setup and management.
    """

    @staticmethod
    def parse_arguments() -> ExperimentArguments:
        """
        Parse command-line arguments for task-arithmetic experiments with PEFT-based LLM fine-tuning for
        personality assessment.
        """
        parser = HfArgumentParser(ExperimentArguments)
        args = parser.parse_args_into_dataclasses()[0]
        if args.use_peft is not None and args.scale_peft is None:
            print(
                "'scale_peft' is required when 'use_peft' is set. Setting to default value of 1.0."
            )
            args.scale_peft = 1.0
        if args.use_peft is None:
            print("'scale_peft' is ignored because 'use_peft' is not set.")
        if args.num_workers is None:
            args.num_workers = Utilities.get_workers()
        return args

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set seed for reproducibility.
        """
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            ":4096:8"  # VITAL: This has to come before torch.backends.cudnn.deterministic = True
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        hf_set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_float32_matmul_precision("medium")
        seed_everything(seed)

    @staticmethod
    def suppress_warnings():
        """
        Suppress warnings from various libraries.
        """
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        warnings.filterwarnings("ignore")
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="TypedStorage is deprecated"
        )
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="torch.distributed.*"
        )
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        warnings.filterwarnings("ignore", category=FutureWarning, module="datasets")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        transformers.logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        pd.options.mode.chained_assignment = None

    @staticmethod
    def get_workers() -> int:
        """
        Get the optimal number of workers for data loading.
        """
        num_cpu_cores = os.cpu_count() if os.cpu_count() is not None else 1
        optimal_workers = min(16, max(1, num_cpu_cores // 2))
        return optimal_workers

    @staticmethod
    def identify_devices() -> List[int]:
        """
        Identify available GPU devices.
        """
        devices: Union[int, List[int], str]
        if torch.cuda.is_available():
            devices = [int(i) for i in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if i.strip().isdigit()]
            if not devices:
                devices = list(range(torch.cuda.device_count()))
            print(f"CUDA devices available: {devices}")
        else:
            devices = []
            print("No CUDA devices are available.")
        #     devices = list(range(torch.cuda.device_count()))
        #     # print(f"CUDA devices available: {devices}")
        # else:
        #     devices = []
        #     print("No CUDA devices are available.")
        return devices

    @staticmethod
    def generate_experiment_id(args: ExperimentArguments) -> str:
        """
        Generate a unique identifier for the experiment based on selected arguments.
        """
        dataset_abbr = args.dataset[0].upper() if args.dataset else 'd'
        def abbreviate_split(split):
            if split:
                parts = split.split('-')
                if len(parts) >= 3:
                    abbr = ''.join([part[0] for part in parts[:2]]) + parts[2]
                else:
                    abbr = ''.join([part[0] for part in parts])
                return abbr.lower()
            else:
                return 'split'
        split_abbr = abbreviate_split(args.split)
        model_abbr = args.model_name[0].lower() if args.model_name else 'm'
        seed_str = f"S{args.seed}"
        epochs_str = f"E{args.epochs}"
        id_parts = [dataset_abbr, split_abbr, model_abbr, seed_str, epochs_str]
        if args.use_peft is not None and args.scale_peft is not None:
            scale_peft_str = f"Sp{args.scale_peft}"
            id_parts.append(scale_peft_str)
        if args.subset is not None:
            subset_str = f"Ss{args.subset}"
            id_parts.append(subset_str)
        experiment_id = '-'.join(id_parts)
        return experiment_id

    @staticmethod
    @rank_zero_only
    def save_experiment_metadata(
        args: ExperimentArguments, experiment_id: str, output_dir: str
    ) -> None:
        """
        Save the experiment metadata to a JSON file.
        """
        metadata_path = os.path.join(output_dir, "experiment_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    metadata = {}
        else:
            metadata = {}
        metadata[experiment_id] = vars(args)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    @rank_zero_only
    def update_experiment_metadata(
        output_dir: str, experiment_id: str, results: dict
    ) -> None:
        """
        Update the experiment metadata with results.
        """
        metadata_path = os.path.join(output_dir, "experiment_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    metadata = {}
        else:
            raise FileNotFoundError(f"Experiment metadata not found at {metadata_path}")
        if experiment_id in metadata:
            metadata[experiment_id]["results"] = Utilities.sanitize_results(results)
        else:
            metadata[experiment_id] = {"results": Utilities.sanitize_results(results)}
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"Experiment metadata updated with results at {metadata_path}")

    @staticmethod
    @rank_zero_only
    def save_experiment_results(
        output_dir: str, experiment_id: str, results: dict
    ) -> None:
        """
        Save the experiment results to a JSON file.
        """
        results = Utilities.sanitize_results(results)
        results_path = os.path.join(output_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: {results_path} is corrupted. Overwriting with new data."
                    )
                    existing_results = {}
        else:
            existing_results = {}
        existing_results[experiment_id] = results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=4)
        print(f"Experiment results saved to {results_path}")

    @staticmethod
    def sanitize_results(data):
        """
        Sanitize the results to ensure JSON compatibility.
        """
        handlers = {
            dict: lambda d: {k: Utilities.sanitize_results(v) for k, v in d.items()},
            list: lambda d: [Utilities.sanitize_results(v) for v in d],
            float: lambda d: (
                ("Infinity" if d > 0 else "-Infinity" if math.isinf(d) else "NaN")
                if math.isnan(d) or math.isinf(d)
                else d
            ),
        }
        return handlers.get(type(data), lambda d: d)(data)

    @staticmethod
    def find_max_batch_size(
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        args: ExperimentArguments,
    ) -> int:
        """
        Find the maximum batch size that can fit on the GPU.
        """
        model.to(device)
        model.eval()
        batch_size = args.batch_size
        dummy_text = "Hello World!"
        encoded_text = tokenizer(
            dummy_text,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"].to(device)
        attention_mask = encoded_text["attention_mask"].to(device)
        with torch.no_grad():
            while True:
                try:
                    input_ids_batch = input_ids.repeat(batch_size, 1)
                    attention_mask_batch = attention_mask.repeat(batch_size, 1)
                    outputs = model(
                        input_ids=input_ids_batch, attention_mask=attention_mask_batch
                    )
                    if outputs is not None:
                        batch_size *= 2
                except RuntimeError:
                    batch_size //= 2
                    break
        print(f"Maximum batch size: {batch_size}")
        return batch_size

    @staticmethod
    def find_optimal_lr(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        args: ExperimentArguments,
    ) -> float:
        """
        Find the optimal learning rate using the learning rate finder.
        """
        initial_lr = args.lr
        output_dir_for_tuner = os.path.join(args.output, "lr_finder")
        trainer = pl.Trainer(
            default_root_dir=output_dir_for_tuner,
            max_epochs=1,
            accumulate_grad_batches=int(args.grad_steps),
            accelerator=args.accelerator,
            devices=args.devices,
            precision=16,
            gradient_clip_val=0.5,
            # deterministic=True                # Doesn't work with Tuner for some reason
        )
        tuner = Tuner(trainer)
        try:
            lr_finder = tuner.lr_find(model, train_loader)
            if lr_finder is not None:
                best_lr = lr_finder.suggestion()
                if best_lr is None:
                    raise ValueError(
                        "Learning rate finder failed to find a suggestion."
                    )
        except Exception as e:
            print(f"Learning rate finder failed with error: {e}")
            best_lr = initial_lr
        print(f"Optimal LR: {best_lr}")
        return best_lr

    @staticmethod
    def clear_cuda_memory() -> None:
        """
        Clear CUDA memory and reset peak memory stats.
        """
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def load_env(dotenv_path: str) -> None:
        """
        Load environment variables from a .env file.
        """
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            if os.getenv("WANDB_API_TOKEN"):
                print("\n")
                # print("WandB API token loaded successfully.")
            else:
                print("WandB API token not found.")
        else:
            print(
                f".env file not found at {dotenv_path}. Please ensure the correct path."
            )

    @staticmethod
    def create_loggers(
        output_dir: str, experiment_id: str, args: ExperimentArguments
    ) -> tuple:
        """
        Create logging objects for TensorBoard and Weights & Biases.
        """
        tb_name = "tb_logs"
        wandb_project = "Master's Thesis"
        tb_logger = TensorBoardLogger(save_dir=output_dir, name=tb_name)
        wandb_logger = WandbLogger(
            project=wandb_project,
            save_dir=output_dir,
            name=experiment_id,
        )

        @rank_zero_only
        def log_wandb_hparams() -> None:
            wandb_logger.log_hyperparams(
                {
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "seed": args.seed,
                }
            )

        log_wandb_hparams()
        return tb_logger, wandb_logger

    @staticmethod
    def create_callbacks(output_dir: str, args: ExperimentArguments) -> List[Any]:
        """
        Create model checkpoint and early stopping callbacks.
        """
        best_model_checkpoint = ModelCheckpoint(
            dirpath=output_dir,
            filename="best-model-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        )
        epoch_checkpoint = ModelCheckpoint(
            dirpath=output_dir,
            filename="epoch-{epoch:02d}",
            save_top_k=-1,
            save_weights_only=False,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=3, mode="min"
        )
        swa_callback = StochasticWeightAveraging(
            swa_lrs=args.lr * 5,
            swa_epoch_start=1,
            annealing_epochs=1,
            annealing_strategy="cos",
        )
        return [
            best_model_checkpoint,
            epoch_checkpoint,
            early_stopping_callback,
            swa_callback,
        ]

    @staticmethod
    def set_paths(args: ExperimentArguments) -> tuple:
        """
        Set the paths for the experiment.
        """
        output_dir = args.output
        experiment_id = Utilities.generate_experiment_id(args)
        print(f"Experiment ID: {experiment_id}")
        os.makedirs(output_dir, exist_ok=True)
        Utilities.save_experiment_metadata(args, experiment_id, output_dir)
        output_dir_exp = os.path.join(output_dir, experiment_id)
        os.makedirs(output_dir_exp, exist_ok=True)
        args.base_output_dir = output_dir
        args.output = output_dir_exp
        return output_dir_exp, experiment_id, args

    @staticmethod
    def print_trainable_params(model: torch.nn.Module) -> None:
        """
        Print the number of trainable parameters.
        """
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_trainable_params}")
        lora_text = "lora" if "lora" in model.state_dict() else "LoRA"
        if lora_text in model.state_dict():
            lora_params = sum(p.numel() for p in model.state_dict()[lora_text].values())
            print(f"Total trainable {lora_text} parameters: {lora_params}")

    @staticmethod
    def refine_hparams(args: ExperimentArguments) -> ExperimentArguments:
        """
        Refine hyperparameters before training.
        """
        check_data_manager = DataManager(args)
        check_data_manager.setup()
        tokenizer = check_data_manager.tokenizer
        check_train_loader = check_data_manager.train_dataloader()
        model_hparams = Namespace(
            lr=args.lr,
            epochs=args.epochs,
            warmup_steps=1,
            accumulate_grad_batches=int(args.grad_steps),
        )
        model: LightningModule = CLMModel(
            args.model_name,
            model_hparams,
            args.use_peft,
            args.scale_peft,
            tokenizer,
        )
        optimal_lr = Utilities.find_optimal_lr(model, check_train_loader, args)
        args.optimal_lr = optimal_lr
        return args

    @staticmethod
    def housekeep() -> Tuple[ExperimentArguments, List[Any], List[Any], Dict[str, Any]]:
        """
        Perform housekeeping tasks before training.
        """
        args = Utilities.parse_arguments()
        dotenv_path = os.path.expanduser("./.env")
        Utilities.load_env(dotenv_path)
        Utilities.set_seed(args.seed)
        devices = Utilities.identify_devices()
        if not devices:
            multi_gpu = False
            devices = 1
            args.devices = 1
            accelerator = "cpu"
            args.accelerator = accelerator
        elif len(devices) == 1:
            multi_gpu = False
            accelerator = "gpu"
            args.accelerator = accelerator
            args.devices = devices
        else:
            multi_gpu = len(devices) > 1
            accelerator = "auto"
            args.accelerator = accelerator
            args.devices = devices
        ava_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        device_config = {
            "devices": devices,
            "multi_gpu": multi_gpu,
            "current_device": ava_device,
            "num_workers": args.num_workers,
        }
        args = Utilities.refine_hparams(args)
        output_dir_exp, exp_id, args = Utilities.set_paths(args)
        args.exp_id = exp_id
        tb_logger, wandb_logger = Utilities.create_loggers(output_dir_exp, exp_id, args)
        callbacks = Utilities.create_callbacks(output_dir_exp, args)
        loggers = [tb_logger, wandb_logger]
        callbacks = list(callbacks)
        Utilities.clear_cuda_memory()
        return args, loggers, callbacks, device_config
