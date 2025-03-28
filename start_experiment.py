# start_experiment.py
"""
Script to initialize and run an experiment, including data preparation, model
training, and evaluation.
"""
import os

import pytorch_lightning as pl
import torch

import experiment_config
from src.utils.main import Utilities
from src.utils.pipeline import (calculate_warmup_steps, initialize_model,
                                perform_evaluation, prepare_data,
                                setup_experiment)

Utilities.suppress_warnings()


def main() -> None:
    """
    Main function to run the experiment, including setup, training, evaluation,
    and saving results.
    """
    # Set up experiment
    args, loggers, callbacks, device_config = setup_experiment()
    data_manager = prepare_data(args)
    tokenizer = data_manager.tokenizer
    warmup_steps = calculate_warmup_steps(args, data_manager, device_config)

    # Initialize model
    model = initialize_model(args, data_manager, warmup_steps)
    # print(model)
    # print(f"===" * 20)
    # Utilities.print_trainable_params(model)
    # print(f"===" * 20)

    # Define Psychometric Evaluation parameters
    if args.dataset == "pandora":   # Extract OCEAN traits using BFI-10
        sample_question = experiment_config.sample_question_bfi10
        possible_answers = experiment_config.possible_answers_bfi10
    elif args.dataset == "emotion":  # Extract emotions using PANAS-X
        sample_question = experiment_config.sample_question_panas_x
        possible_answers = experiment_config.sample_answers_panas_x
    else:
        raise ValueError("Invalid dataset specified. Cannot identify evaluation method.")
    peft_scales = experiment_config.peft_scales
    temperatures = experiment_config.temperatures

    # Psychometric Evaluation (Pre-Finetuning)
    if args.use_peft:   # Using PEFT
        custom_eval_pre = {}
        for scale in peft_scales:
            personality_eval = perform_evaluation(model, tokenizer, temperatures,
                                                  sample_question, possible_answers, args, scale)
            custom_eval_pre[f"scale_{scale}"] = personality_eval.to_dict(
                orient="records"
            )
    else:   # Base model
        personality_eval = perform_evaluation(model, tokenizer, temperatures,
                                                sample_question, possible_answers, args)
        custom_eval_pre = {}
        custom_eval_pre["scale_None"] = personality_eval.to_dict(orient="records")

    # Fine-Tuning
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
    trainer.fit(model, train_dataloaders=data_manager.train_dataloader(),
                val_dataloaders=data_manager.val_dataloader())
    train_metrics = trainer.callback_metrics

    # Testing
    model.eval()
    test_results = trainer.test(model, dataloaders=data_manager.test_dataloader())

    # Psychometric Evaluation (Post-Finetuning)
    if args.use_peft:   # Using PEFT
        custom_eval_post = {}
        for scale in peft_scales:
            personality_eval = perform_evaluation(model, tokenizer, temperatures,
                                                  sample_question, possible_answers,
                                                  args, scale)
            custom_eval_post[f"scale_{scale}"] = personality_eval.to_dict(
                orient="records"
            )
    else:   # Base model
        personality_eval = perform_evaluation(model, tokenizer, temperatures,
                                                   sample_question, possible_answers, args)
        custom_eval_post = {}
        custom_eval_post["scale_None"] = personality_eval.to_dict(orient="records")

    # Compile & Save Results
    results = {}
    results["train_metrics"] = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in train_metrics.items()
    }
    if test_results:
        test_metrics = test_results[0]
        results["test_metrics"] = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in test_metrics.items()
        }
    results["custom_eval_pre"] = custom_eval_pre
    results["custom_eval_post"] = custom_eval_post
    Utilities.save_experiment_results(args.output, args.exp_id, results)
    Utilities.update_experiment_metadata(args.output, args.exp_id, results)


if __name__ == "__main__":
    main()
