# src/utils/args_parser.py
"""
Module defining the ExperimentArguments dataclass for parsing
command-line arguments.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentArguments:
    """
    Dataclass containing configuration arguments for running experiments.

    Attributes:
        dataset (str): Dataset to use (e.g., 'pandora', 'jigsaw').
        split (Optional[str]): Dataset split to use.
        subset (Optional[int]): Number of samples to subset.
        output (str): Directory for experiment outputs.
        model_name (str): Pre-trained model to use.
        seed (int): Seed for reproducibility.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        grad_steps (int): Gradient accumulation steps.
        use_peft (Optional[str]): PEFT method to use.
        warmup_ratio (float): Warmup ratio for the learning rate scheduler.
        num_workers (Optional[int]): Number of CPU workers for data loading.
        accelerator (Optional[str]): Accelerator type ('cpu', 'gpu', 'auto').
        devices (Optional[str]): Devices to use for training.
        optimal_lr (Optional[float]): Optimal learning rate found by the LR finder.
        exp_id (Optional[str]): Unique experiment identifier.
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
                "'openness-bot-5', ... | Different splits across datasets"
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
    devices: Optional[str] = field(
        default=None,
        metadata={
            "help": "Devices to be used for training. Can be an int or a list of ints."
        },
    )
    optimal_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optimal learning rate to be found by the Learning Rate Finder"
        },
    )
    exp_id: Optional[str] = field(
        default=None,
        metadata={"help": "Unique ID for the experiment."},
    )
