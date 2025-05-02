# src/utils/pipeline.py
"""
Utility module containing helper functions for the experiment pipeline.
"""
import math
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import torch
from peft.helpers import rescale_adapter_scale
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM

from src.data_manager import DataManager
from src.eval_manager import EvalManager
from src.model_manager import CLMModel
from src.peft_manager import PEFTManager
from src.utils.main import Utilities
from src.eval_results_manager import EvalResultsManager
from experiment_config import (IPIP_INSTRUCTIONS, IPIP_LIKERT_OPTIONS, 
                              IPIP_LIKERT_VALUES, IPIP_ITEMS_FILE)


def setup_experiment() -> tuple:
    """
    Sets up the experiment by parsing arguments and preparing the environment.

    Returns:
        Tuple containing arguments, loggers, callbacks, and device configuration.
    """
    args, loggers, callbacks, device_config = Utilities.housekeep()
    return args, loggers, callbacks, device_config


def prepare_data(args: Namespace) -> DataManager:
    """
    Prepares the data using the DataManager.

    Args:
        args: Experiment arguments.

    Returns:
        DataManager: The initialized data manager with prepared data.
    """
    data_manager = DataManager(args)
    data_manager.setup(args)
    return data_manager


def initialize_model(args: Namespace, data_manager: DataManager,
                     warmup_steps: int) -> CLMModel:
    """
    Initializes the model and wraps it with PEFT if specified.

    Args:
        args: Experiment arguments.
        data_manager: DataManager with tokenizer.
        warmup_steps: Number of warmup steps for the scheduler.

    Returns:
        CLMModel: The initialized language model.
    """
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    pretrained_model.resize_token_embeddings(len(data_manager.tokenizer))

    if not args.use_peft:
        for param in pretrained_model.parameters():
            param.requires_grad = True
    else:
        pretrained_model = PEFTManager.apply_peft(pretrained_model, args.use_peft)

    model_hparams = Namespace(
        lr=args.lr,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        accumulate_grad_batches=int(args.grad_steps),
    )

    clm_model = CLMModel(
        pretrained_model,
        model_hparams,
    )
    return clm_model


def calculate_warmup_steps(args: Namespace, data_manager: DataManager,
                           device_config: Dict[str, Any]) -> int:
    """
    Calculates the number of warmup steps for the scheduler.

    Args:
        args: Experiment arguments.
        data_manager: DataManager with tokenizer.
        device_config: Device configuration.

    Returns:
        int: Number of warmup steps.
    """
    len_train_loader = len(data_manager.train_dataloader())
    total_steps = (
        math.ceil(len_train_loader / args.grad_steps) * args.epochs
        if not device_config["multi_gpu"]
        else math.ceil(
            (len_train_loader * args.epochs * len(device_config["devices"]))
            / args.grad_steps
        )
    )
    warmup_steps = max(1, int(total_steps * min(max(args.warmup_ratio, 0.03), 0.1)))
    return warmup_steps


def perform_evaluation(model, tokenizer, temperatures,
                       sample_question, possible_answers,
                       args, scale_peft=None) -> pd.DataFrame:
    """
    Performs evaluation of the model using the specified question and answers.
    Supports different evaluation types based on the dataset.

    Args:
        model: The language model to evaluate.
        tokenizer: Tokenizer corresponding to the model.
        temperatures: List of temperatures for sampling.
        sample_question: The question prompt to evaluate.
        possible_answers: List of possible answers to the question.
        args: Experiment arguments.
        scale_peft: Scale for PEFT tuning.

    Returns:
        pd.DataFrame: DataFrame containing probabilities for each answer.
    """
    # Handle PEFT scaling if applicable
    if (scale_peft is not None and args.use_peft
        and any(isinstance(module, LoraLayer) for module in model.modules())):
        with rescale_adapter_scale(model, scale_peft):
            eval_results = EvalManager.extract_answers(
                model,
                tokenizer,
                sample_question,
                possible_answers,
                temps=temperatures,
            )
    else:
        eval_results = EvalManager.extract_answers(
            model,
            tokenizer,
            sample_question,
            possible_answers,
            temps=temperatures,
        )
    
    return eval_results


def perform_ipip120_evaluation(model, tokenizer, args, scale_peft=None) -> pd.DataFrame:
    """
    Performs IPIP-120 personality evaluation.
    
    Args:
        model: The language model to evaluate.
        tokenizer: Tokenizer corresponding to the model.
        args: Experiment arguments.
        scale_peft: Scale for PEFT tuning.
        
    Returns:
        pd.DataFrame: DataFrame containing IPIP-120 evaluation results.
    """
    # Handle PEFT scaling if applicable
    if (scale_peft is not None and args.use_peft
        and any(isinstance(module, LoraLayer) for module in model.modules())):
        with rescale_adapter_scale(model, scale_peft):
            ipip_results = EvalManager.extract_ipip120(
                model,
                tokenizer,
                IPIP_ITEMS_FILE,
                IPIP_INSTRUCTIONS,
                IPIP_LIKERT_OPTIONS,
                IPIP_LIKERT_VALUES
            )
    else:
        ipip_results = EvalManager.extract_ipip120(
            model,
            tokenizer,
            IPIP_ITEMS_FILE,
            IPIP_INSTRUCTIONS,
            IPIP_LIKERT_OPTIONS,
            IPIP_LIKERT_VALUES
        )
    
    return ipip_results
