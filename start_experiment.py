# start_experiment.py
"""
Script to initialize and run an experiment, including data preparation, model
training, and evaluation.
"""
import math
from argparse import Namespace

import pandas as pd
import pytorch_lightning as pl
import torch
from peft.helpers import rescale_adapter_scale
from transformers import AutoModelForCausalLM

from src.data_manager import DataManager
from src.eval_manager import EvalManager
from src.model_manager import CLMModel
from src.peft_manager import PEFTManager
from src.utils.main import Utilities

Utilities.suppress_warnings()


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


# def initialize_model(args, data_manager, warmup_steps):
def initialize_model(
    args: Namespace, data_manager: DataManager, warmup_steps: int
) -> CLMModel:
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

    # Apply PEFT if specified
    if args.use_peft:
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


def perform_evaluation(
    model,
    tokenizer,
    temperatures,
    sample_question,
    possible_answers,
    args,
    scale_peft=None,
) -> pd.DataFrame:
    """
    Performs model evaluation by generating responses to specific questions.

    Args:
        model: The trained model.
        tokenizer: Tokenizer corresponding to the model.
        temperatures: List of temperatures for sampling.
        sample_question: The question prompt.
        possible_answers: List of possible answers.
        args: Experiment arguments.
        scale_peft: Scaling factor for PEFT layers.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if scale_peft and args.use_peft and hasattr(model, args.use_peft):
        with rescale_adapter_scale(model, scale_peft):
            with torch.no_grad():
                personality_eval_results = EvalManager.extract_answers(
                    model,
                    tokenizer,
                    sample_question,
                    possible_answers,
                    temps=temperatures,
                )
    else:
        with torch.no_grad():
            personality_eval_results = EvalManager.extract_answers(
                model, tokenizer, sample_question, possible_answers, temps=temperatures
            )
    return personality_eval_results


def main() -> None:
    """
    Main function to run the experiment, including setup, training, evaluation,
    and saving results.
    """
    # Setup
    args, loggers, callbacks, device_config = setup_experiment()
    data_manager = prepare_data(args)
    tokenizer = data_manager.tokenizer

    # Calculate warmup steps
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

    # Initialize model
    model = initialize_model(args, data_manager, warmup_steps)
    # print("Model initialized")
    # print(model)

    # Define Evaluation Parameters
    temperatures = [0.7, 0.8, 0.9, 1]  # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
    sample_question = "I see myself as someone who"
    possible_answers = [
        "is reserved.",
        "is generally trusting.",
        "tends to be lazy.",
        "is relaxed, handles stress well.",
        "has few artistic interests.",
        "is outgoing, sociable.",
        "tends to find fault with others.",
        "does a thorough job.",
        "gets nervous easily.",
        "has an active imagination.",
    ]

    # Pre-Training Evaluation
    personality_eval_pre = perform_evaluation(
        model, tokenizer, temperatures, sample_question, possible_answers, args
    )

    # Training
    model.train()
    trainer = pl.Trainer(
        default_root_dir=args.output,
        max_epochs=args.epochs,
        accumulate_grad_batches=int(args.grad_steps),
        accelerator=args.accelerator,
        devices=args.devices,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=0.5,
        val_check_interval=0.5,
        precision=16,
        deterministic=True,
    )
    trainer.fit(
        model,
        train_dataloaders=data_manager.train_dataloader(),
        val_dataloaders=data_manager.val_dataloader(),
    )

    # Collect Training Results
    results = {}
    train_metrics = trainer.callback_metrics
    results["train_metrics"] = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in train_metrics.items()
    }

    # Testing
    test_results = trainer.test(model, dataloaders=data_manager.test_dataloader())
    if test_results:
        test_metrics = test_results[0]
        results["test_metrics"] = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in test_metrics.items()
        }

    # Post-Training Evaluation with variable PEFT scales
    peft_scales = [
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        2.5,
        5.0,
        7.5,
        10.0,
        12.5,
        15.0,
        20.0,
        25.0,
        40.0,
        50.0,
        75.0,
        100.0,
        150.0,
        200.0,
    ]
    personality_eval_post = {}
    for scale in peft_scales:
        personality_eval = perform_evaluation(
            model,
            tokenizer,
            temperatures,
            sample_question,
            possible_answers,
            args,
            scale,
        )
        personality_eval_post[f"scale_{scale}"] = personality_eval.to_dict(
            orient="records"
        )

    # Save Results
    personality_eval_pre_dict = personality_eval_pre.to_dict(orient="records")
    results["personality_eval_pre"] = personality_eval_pre_dict
    results["personality_eval_post"] = personality_eval_post
    Utilities.save_experiment_results(args.output, args.exp_id, results)
    Utilities.update_experiment_metadata(args.output, args.exp_id, results)


if __name__ == "__main__":
    main()
