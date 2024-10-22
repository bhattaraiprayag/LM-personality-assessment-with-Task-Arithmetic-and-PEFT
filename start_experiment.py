# start_experiment.py

"""
Module to initiate and run the personality assessment experiment.
"""

import math
from argparse import Namespace

import pytorch_lightning as pl
import torch

from src.data_manager import DataManager
from src.eval_manager import EvalManager
from src.model_manager import CLMModel
from src.utilities import Utilities
Utilities.suppress_warnings()


def setup_experiment():
    """
    Perform housekeeping tasks and set up the experiment environment.
    """
    args, loggers, callbacks, device_config = Utilities.housekeep()
    return args, loggers, callbacks, device_config


def prepare_data(args):
    """
    Set up the data manager.
    """
    data_manager = DataManager(args)
    data_manager.setup(args)
    return data_manager


def initialize_model(args, data_manager, warmup_steps):
    """
    Initialize the language model.
    """
    tokenizer = data_manager.tokenizer
    model_hparams = Namespace(
        lr=args.lr,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        accumulate_grad_batches=int(args.grad_steps),
    )
    model = CLMModel(
        args.model_name,
        model_hparams,
        args.use_peft,
        args.scale_peft,
        tokenizer,
    )
    return model


def perform_evaluation(model, tokenizer, temperatures, sample_question, possible_answers):
    """
    Perform personality evaluation before and after training.
    """
    model.eval()
    with torch.no_grad():
        personality_eval_results = EvalManager.extract_answers(
            model, tokenizer, sample_question, possible_answers, temps=temperatures
        )
    return personality_eval_results


def main():
    """
    Main function to set up and execute the experiment.
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
    
    # Define Evaluation Parameters
    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
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
        model, tokenizer, temperatures, sample_question, possible_answers
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
    
    # Post-Training Evaluation
    personality_eval_post = perform_evaluation(
        model, tokenizer, temperatures, sample_question, possible_answers
    )
    
    # Save Results
    personality_eval_pre_dict = personality_eval_pre.to_dict(orient="records")
    personality_eval_post_dict = personality_eval_post.to_dict(orient="records")
    results["personality_eval_pre"] = personality_eval_pre_dict
    results["personality_eval_post"] = personality_eval_post_dict
    Utilities.save_experiment_results(args.output, args.exp_id, results)
    Utilities.update_experiment_metadata(args.base_output_dir, args.exp_id, results)


if __name__ == "__main__":
    main()
