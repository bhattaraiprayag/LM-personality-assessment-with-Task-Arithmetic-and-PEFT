# start_experiment.py
"""
Script to initialize and run an experiment, including data preparation, model
training, and evaluation.
"""
import os
from src.utils.pipeline import (
    setup_experiment,
    prepare_data,
    initialize_model,
    calculate_warmup_steps,
    perform_evaluation,
)
import pytorch_lightning as pl
import torch
from src.utils.main import Utilities
import experiment_config

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

    # Define personality evaluation parameters
    temperatures = experiment_config.temperatures
    sample_question = experiment_config.sample_question
    possible_answers = experiment_config.possible_answers
    peft_scales = experiment_config.peft_scales

    # Personality Evaluation (Pre-Fine-Tuning)
    if args.use_peft:   # Using PEFT
        personality_eval_pre = {}
        for scale in peft_scales:
            personality_eval = perform_evaluation(model, tokenizer, temperatures,
                                                  sample_question, possible_answers, args, scale)
            personality_eval_pre[f"scale_{scale}"] = personality_eval.to_dict(
                orient="records"
            )
    else:   # Base model
        personality_eval = perform_evaluation(model, tokenizer, temperatures,
                                                sample_question, possible_answers, args)
        personality_eval_pre = {}
        personality_eval_pre["scale_None"] = personality_eval.to_dict(orient="records")

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

    # Personality Evaluation (Post-Fine-Tuning)
    if args.use_peft:   # Using PEFT
        personality_eval_post = {}
        for scale in peft_scales:
            personality_eval = perform_evaluation(model, tokenizer, temperatures,
                                                  sample_question, possible_answers,
                                                  args, scale)
            personality_eval_post[f"scale_{scale}"] = personality_eval.to_dict(
                orient="records"
            )
    else:   # Base model
        personality_eval = perform_evaluation(model, tokenizer, temperatures,
                                                   sample_question, possible_answers, args)
        personality_eval_post = {}
        personality_eval_post["scale_None"] = personality_eval.to_dict(orient="records")

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
    results["personality_eval_pre"] = personality_eval_pre
    results["personality_eval_post"] = personality_eval_post
    Utilities.save_experiment_results(args.output, args.exp_id, results)
    Utilities.update_experiment_metadata(args.output, args.exp_id, results)


if __name__ == "__main__":
    main()
