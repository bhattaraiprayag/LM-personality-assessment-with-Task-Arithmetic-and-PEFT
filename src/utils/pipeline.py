# src/utils/pipeline.py
"""
Utility module containing helper functions for the experiment pipeline.
"""
import math
import pandas as pd
import torch

from argparse import Namespace
from typing import Any, Dict, List, Tuple, Optional, Union
from peft.helpers import rescale_adapter_scale
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM

from src.data_manager import DataManager
from src.eval_manager import EvalManager
from src.model_manager import CLMModel
from src.peft_manager import PEFTManager
from src.utils.main import Utilities
from src.eval_results_manager import EvalResultsManager


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

def run_inventory_eval(
    model,
    tok,
    args,
    phase: str,
    *,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    scale: Optional[Union[float, str]] = None,
) -> None:
    """
    Runs `score_likert` for whichever inventories belong to `args.dataset`
    (Pandora → BFI10 + IPIP120, Emotion → PANASX) and streams the rows
    into CSV(s) via `EvalResultsManager.append_rows`.
    """
    # Handle PEFT scaling if applicable
    if (scale is not None and args.use_peft
        and any(isinstance(module, LoraLayer) for module in model.modules())):
        with rescale_adapter_scale(model, scale):
            _run_inventory_eval_inner(model, tok, args, phase, epoch, step, scale)
    else:
        _run_inventory_eval_inner(model, tok, args, phase, epoch, step, scale)


def _run_inventory_eval_inner(
    model,
    tok,
    args,
    phase: str,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    scale: Optional[Union[float, str]] = None,
) -> None:
    """Helper function to actually run the evaluations after PEFT context is set up."""
    mgr = EvalManager(model=model, tokenizer=tok)
    invs = ["PANASX"] if args.dataset == "emotion" else ["BFI10", "IPIP120"]
    for inv in invs:
        df = mgr.score_likert(inventory_name=inv)
        EvalResultsManager.append_rows(
            df=df,
            phase=phase,
            inventory=inv,
            exp_id=args.exp_id,
            output_dir=args.output,
            use_peft=("lora" if args.use_peft else "baseline"),
            lora_scale=("baseline" if not scale else scale),
            epoch=epoch,
            step=step,
        )
