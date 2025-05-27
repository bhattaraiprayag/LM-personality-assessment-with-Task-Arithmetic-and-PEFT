# src/utils/main.py
"""
Main utility module containing functions for argument parsing, environment
setup, and housekeeping tasks.
"""
import json
import logging
import math
import os
import random
import warnings
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Union

import datasets
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from dotenv import load_dotenv
from lightning.pytorch import seed_everything
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoModelForCausalLM, HfArgumentParser
from transformers import set_seed as hf_set_seed

from src.data_manager import DataManager
from src.model_manager import CLMModel
from src.utils.args_parser import ExperimentArguments
from src.utils.helper import print_output


class Utilities:
    """
    Class providing static utility methods for setting up experiments, parsing arguments,
    and managing devices.
    """

    @staticmethod
    def save_json(filepath: str, data: Any) -> None:
        """
        Saves data to a JSON file.

        Args:
            filepath (str): Path to save the JSON file
            data (Any): Data to save (must be JSON serializable)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def parse_arguments() -> ExperimentArguments:
        """
        Parses command-line arguments using HfArgumentParser.

        Returns:
            ExperimentArguments: Parsed arguments.
        """
        parser = HfArgumentParser(ExperimentArguments)
        args = parser.parse_args_into_dataclasses()[0]
        if args.use_peft is None:
            print_output("'scale_peft' is ignored because 'use_peft' is not set.")
        if args.num_workers is None:
            args.num_workers = Utilities.get_workers()
        return args

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Sets the seed for reproducibility across runs.

        Args:
            seed (int): Seed value.
        """
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            ":4096:8"  #  CAUTION: This has to come before torch.backends.cudnn.deterministic = True
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
    def suppress_warnings() -> None:
        """
        Suppresses warnings and sets logging verbosity to error.
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
        Sets the optimal number of workers for DataLoader.

        Returns:
            int: Number of workers.
        """
        num_cpu_cores = os.cpu_count() or 1
        optimal_workers = min(16, max(1, num_cpu_cores // 2))
        return optimal_workers

    @staticmethod
    def identify_gpu_devices(args: ExperimentArguments) -> List[int]:
        """
        Identifies available CUDA devices.

        Returns:
            Union[int, List[int], str]: List of device IDs.
        """
        devices: Union[int, List[int], str]
        devices_arg = args.devices
        if torch.cuda.is_available():
            if devices_arg is not None:
                devices_str_list = [
                    d.strip() for d in devices_arg.split(",") if d.strip().isdigit()
                ]
                if len(devices_str_list) == 1:
                    devices = [int(devices_str_list[0])]
                elif devices_str_list:
                    devices = [int(d) for d in devices_str_list]
                else:
                    devices = list(range(torch.cuda.device_count()))
            else:
                devices = list(range(torch.cuda.device_count()))
            print_output(f"CUDA devices available: {devices}")
        else:
            devices = []
            print_output("No CUDA devices are available.")
        return devices

    @staticmethod
    def generate_experiment_id(args: ExperimentArguments) -> str:
        """
        Generates a unique experiment ID based on the provided arguments.

        Args:
            args (ExperimentArguments): Parsed arguments.

        Returns:
            str: Experiment ID.
        """
        dataset_abbr = args.dataset[0].upper() if args.dataset else "d"

        def abbreviate_split(split):
            if split:
                parts = split.split("-")
                if len(parts) >= 3:
                    abbr = "".join([part[0] for part in parts[:2]]) + parts[2]
                else:
                    abbr = "".join([part[0] for part in parts])
                return abbr.lower()
            else:
                return "split"

        split_abbr = abbreviate_split(args.split)
        model_abbr = args.model_name.lower() if args.model_name else "m"
        seed_str = f"Se-{args.seed}"
        epochs_str = f"Ep-{args.epochs}"
        id_parts = [dataset_abbr, split_abbr, model_abbr, seed_str, epochs_str]

        if args.use_peft is not None:
            use_peft_str = f"Pe-{args.use_peft}"
            id_parts.append(use_peft_str)
        if args.subset is not None:
            subset_str = f"Ss-{args.subset}"
            id_parts.append(subset_str)
        experiment_id = "-".join(id_parts)
        return experiment_id

    @staticmethod
    @rank_zero_only
    def save_experiment_metadata(args: ExperimentArguments, experiment_id: str,
                                 output_dir: str) -> None:
        """
        Saves experiment metadata to a JSON file.

        Args:
            args (ExperimentArguments): Parsed arguments.
            experiment_id (str): Unique experiment ID.
            output_dir (str): Directory to save metadata.
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
    def update_experiment_metadata(output_dir: str, experiment_id: str,
                                   results: dict) -> None:
        """
        Updates experiment metadata with results.

        Args:
            output_dir (str): Directory containing the metadata file.
            experiment_id (str): Unique experiment ID.
            results (dict): Experiment results

        Raises:
            FileNotFoundError: If metadata file is not found.
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
    def save_experiment_results(output_dir: str, experiment_id: str,
                                results: dict) -> None:
        """
        Saves experiment results to a JSON file.

        Args:
            output_dir (str): Directory to save results.
            experiment_id (str): Unique experiment ID.
            results (dict): Experiment results.

        Raises:
            JSONDecodeError: If the results file is corrupted.
        """
        results = Utilities.sanitize_results(results)
        results_path = os.path.join(output_dir, experiment_id, "results.json")
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
    def sanitize_results(data) -> Any:
        """
        Sanitizes results for JSON serialization.

        Args:
            data: Data to sanitize.

        Returns:
            Any: Sanitized data.
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
    def find_max_batch_size(model: torch.nn.Module, tokenizer, device: torch.device,
                            args: ExperimentArguments) -> int:
        """
        Finds the maximum batch size that can be used for training.

        Args:
            model (torch.nn.Module): Model to train.
            tokenizer: Tokenizer for the model.
            device (torch.device): Device to use.
            args (ExperimentArguments): Parsed arguments.

        Returns:
            int: Maximum batch size.
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
        return batch_size

    @staticmethod
    def find_optimal_lr(model: LightningModule, train_loader: torch.utils.data.DataLoader,
                        args: ExperimentArguments) -> float:
        """
        Finds the optimal learning rate using the Learning Rate Finder.

        Args:
            model (LightningModule): Model to train.
            train_loader (torch.utils.data.DataLoader): Training DataLoader.
            args (ExperimentArguments): Parsed arguments.

        Returns:
            float: Optimal learning rate.
        """
        initial_lr = args.lr
        output_dir_for_tuner = os.path.join(args.output, "lr_finder")
        trainer = pl.Trainer(
            default_root_dir=output_dir_for_tuner,
            max_epochs=1,
            accumulate_grad_batches=int(args.grad_steps),
            accelerator=args.accelerator if args.accelerator is not None else "auto",
            devices=args.devices if args.devices is not None else 1,
            precision=16,
            gradient_clip_val=0.5,
            # deterministic=True        # CAUTION: Doesn't work with Tuner for some reason
        )
        tuner = Tuner(trainer)
        try:
            lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
            if lr_finder is not None:
                suggestion = lr_finder.suggestion()
                if suggestion is not None:
                    best_lr = suggestion
                else:
                    best_lr = initial_lr
            else:
                best_lr = initial_lr
        except Exception as e:
            print_output(f"Learning rate finder failed with error: {e}")
            best_lr = initial_lr
        print_output(f"Optimal LR: {best_lr}")
        return best_lr

    @staticmethod
    def clear_cuda_memory() -> None:
        """
        Clears CUDA memory and resets peak memory stats.
        !!! Does not work; need to fix !!!
        """
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def load_env(dotenv_path: str) -> None:
        """
        Loads environment variables from a .env file.

        Args:
            dotenv_path (str): Path to the .env file.
        """
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            if os.getenv("WANDB_API_TOKEN"):
                print("WandB API token loaded successfully.")
            else:
                print("WandB API token not found.")
        else:
            print(
                f".env file not found at {dotenv_path}. Please ensure the correct path."
            )

    @staticmethod
    def create_loggers(output_dir: str, experiment_id: str,
                       args: ExperimentArguments) -> tuple:
        """
        Creates TensorBoard and WandB loggers for experiment tracking.

        Args:
            output_dir (str): Directory to save logs.
            experiment_id (str): Unique experiment ID.
            args (ExperimentArguments): Parsed arguments.

        Returns:
            tuple: Tuple containing the TensorBoard and WandB loggers.
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
        # return wandb_logger

    @staticmethod
    def create_callbacks(output_dir: str, args: ExperimentArguments) -> List[Any]:
        """
        Creates callbacks for model training.

        Args:
            output_dir (str): Directory to save logs.
            args (ExperimentArguments): Parsed arguments.

        Returns:
            List[Any]: List of callbacks.
        """
        # best_model_checkpoint = ModelCheckpoint(
        #     dirpath=output_dir,
        #     filename="best-model-{epoch:02d}-{val_loss:.4f}",
        #     monitor="val_loss",
        #     mode="min",
        #     save_top_k=1,
        #     save_weights_only=False,
        #     save_last=False,
        # )
        # last_checkpoint = ModelCheckpoint(
        #     dirpath=output_dir,
        #     save_weights_only=False,
        #     save_last=True,
        #     save_top_k=0,
        # )
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
            # best_model_checkpoint,
            # last_checkpoint,
            early_stopping_callback,
            swa_callback,
        ]

    @staticmethod
    def set_paths(args: ExperimentArguments) -> tuple:
        """
        Sets the output directory and experiment ID for the experiment.

        Args:
            args (ExperimentArguments): Parsed arguments.

        Returns:
            tuple: Tuple containing the experiment output directory,
                experiment ID, and updated arguments.
        """
        base_output_dir = args.output
        experiment_id = Utilities.generate_experiment_id(args)
        print_output(f"Experiment ID: {experiment_id}")
        os.makedirs(base_output_dir, exist_ok=True)
        exp_out_dir = os.path.join(base_output_dir, experiment_id)
        os.makedirs(exp_out_dir, exist_ok=True)
        args.exp_out_dir = exp_out_dir
        args.exp_id = experiment_id
        Utilities.save_experiment_metadata(args, experiment_id, base_output_dir)
        return exp_out_dir, experiment_id, args

    @staticmethod
    def print_trainable_params(model: torch.nn.Module) -> None:
        """
        Prints the total number of trainable parameters in the model.

        Args:
            model (torch.nn.Module): Model to evaluate.
        """
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print_output(f"Total trainable parameters: {total_trainable_params}")
        lora_text = "lora" if "lora" in model.state_dict() else "LoRA"
        if lora_text in model.state_dict():
            lora_params = sum(p.numel() for p in model.state_dict()[lora_text].values())
            print_output(f"Total trainable {lora_text} parameters: {lora_params}")

    @staticmethod
    def refine_hparams(args: ExperimentArguments) -> ExperimentArguments:
        """
        Refines the learning rate before training.

        Args:
            args (ExperimentArguments): Parsed arguments.

        Returns:
            ExperimentArguments: Updated arguments.
        """
        check_data_manager = DataManager(args)
        check_data_manager.setup()
        tokenizer = check_data_manager.tokenizer
        check_train_loader = check_data_manager.train_dataloader()
        pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        pretrained_model.resize_token_embeddings(len(tokenizer))
        model_hparams = Namespace(
            lr=args.lr,
            epochs=args.epochs,
            warmup_steps=1,
            accumulate_grad_batches=int(args.grad_steps),
        )
        clm_model: LightningModule = CLMModel(
            pretrained_model,
            model_hparams,
        )
        optimal_lr = Utilities.find_optimal_lr(clm_model, check_train_loader, args)
        args.optimal_lr = optimal_lr
        return args

    @staticmethod
    def housekeep() -> Tuple[ExperimentArguments, List[Any], List[Any], Dict[str, Any]]:
        """
        Performs housekeeping tasks to set up the experiment.

        Returns:
            Tuple[ExperimentArguments, List[Any], List[Any], Dict[str, Any]]:
                Tuple containing arguments, loggers, callbacks, and device configuration.
        """
        args = Utilities.parse_arguments()
        dotenv_path = os.path.expanduser("./.env")
        Utilities.load_env(dotenv_path)
        Utilities.set_seed(args.seed)
        devices = Utilities.identify_gpu_devices(args)
        if not devices:
            multi_gpu = False
            devices = 1
            args.devices = 1
            accelerator = "cpu"
            args.accelerator = accelerator
        elif isinstance(devices, list) and len(devices) == 1:
            multi_gpu = False
            accelerator = "gpu"
            args.accelerator = accelerator
            args.devices = devices
        else:
            multi_gpu = isinstance(devices, list) and len(devices) > 1
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
        print_output(f"Device configuration: {device_config}")
        args = Utilities.refine_hparams(args)
        output_dir_exp, exp_id, args = Utilities.set_paths(args)
        tb_logger, wandb_logger = Utilities.create_loggers(output_dir_exp, exp_id, args)
        callbacks = Utilities.create_callbacks(output_dir_exp, args)
        loggers = [tb_logger, wandb_logger]
        callbacks = list(callbacks)
        args.exp_out_dir = output_dir_exp
        # Utilities.clear_cuda_memory()        # CAUTION: Does not work; need to fix.
        return args, loggers, callbacks, device_config
