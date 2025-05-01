# start_experiment.py
"""
Script to initialize and run an experiment, including data preparation, model
training, and evaluation.
"""
import os

import pytorch_lightning as pl
import torch
from peft import get_peft_model_state_dict

import experiment_config
from src.utils.main import Utilities
from src.utils.pipeline import (calculate_warmup_steps, initialize_model,
                                perform_evaluation, prepare_data,
                                setup_experiment)
from src.utils.callbacks import MidEpochCheckpointCallback
from src.eval_results_manager import EvalResultsManager
from src.utils.perplexity import benchmark_model_perplexity, save_perplexity_results

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

    # Define Psychometric Evaluation parameters based on dataset
    eval_type = EvalResultsManager.get_evaluation_inventory(args.dataset)
    
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

    # Calculate total steps for mid-epoch checkpoints
    total_steps = len(data_manager.train_dataloader()) * args.epochs // int(args.grad_steps)

    # Add mid-epoch checkpoint callback if using PEFT
    if args.use_peft:
        mid_epoch_callback = MidEpochCheckpointCallback(
            args=args,
            tokenizer=tokenizer,
            sample_question=sample_question,
            possible_answers=possible_answers,
            temperatures=temperatures,
            peft_scales=peft_scales,
            total_steps=total_steps,
            eval_type=eval_type
        )
        callbacks.append(mid_epoch_callback)

    # Note: Pre-finetuning evaluation is skipped as per requirements
    # Results dict will only contain training and testing metrics
    results = {}

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

    # Run perplexity benchmarks
    # Extract the training perplexity from the model metrics
    clm_perplexity = {}
    if hasattr(model, 'metrics') and 'val_perplexity' in model.metrics:
        clm_perplexity = {
            'clm_perplexity': model.metrics['val_perplexity'][-1] if model.metrics['val_perplexity'] else None
        }
    
    # Benchmark perplexity
    if args.use_peft:  # LoRA model
        benchmark_results = benchmark_model_perplexity(
            model=model.model,  # Access the wrapped model
            tokenizer=tokenizer,
            seed=args.seed,
            is_lora=True,
            lora_scales=peft_scales
        )
        # Save results to CSV
        perplexity_csv_path = save_perplexity_results(
            output_dir=args.output,
            experiment_id=args.exp_id,
            results=benchmark_results,
            is_lora=True
        )
        # Add reference to the perplexity CSV file
        perplexity_ref = {
            'perplexity_benchmarks': perplexity_csv_path,
            **clm_perplexity
        }
    else:  # Vanilla model
        benchmark_results = benchmark_model_perplexity(
            model=model.model,  # Access the wrapped model
            tokenizer=tokenizer,
            seed=args.seed,
            is_lora=False
        )
        # Save directly to results
        perplexity_ref = {
            'perplexity_benchmarks': benchmark_results,
            **clm_perplexity
        }

    # Psychometric Evaluation (Post-Finetuning)
    custom_eval_post = {}
    if args.use_peft:   # Using PEFT
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
    
    # Save post-finetuning custom evaluation results to CSV
    EvalResultsManager.save_custom_eval_results(
        output_dir=args.output,
        experiment_id=args.exp_id,
        phase="post",
        eval_type=eval_type,
        question=sample_question,
        answers=possible_answers,
        results=custom_eval_post
    )

    # Save final model
    if args.use_peft:
        # Save only LoRA adapter weights
        lora_state_dict = get_peft_model_state_dict(model.model)  # Access the wrapped model
        torch.save(lora_state_dict, os.path.join(args.exp_out_dir, "lora_final.pt"))
    else:
        # Save full model
        model.model.save_pretrained(args.exp_out_dir)  # Access the wrapped model

    # Compile & Save Results (Only training and test metrics, no custom eval)
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
    
    # Add perplexity benchmark results
    results["perplexity"] = perplexity_ref
    
    # Add references to the custom evaluation files in the metadata
    evals_dir = os.path.join(args.output, args.exp_id, "evals")
    results["custom_eval_files"] = {
        "post": os.path.join(evals_dir, f"custom_eval_{eval_type}_post.csv"),
    }
    
    if args.use_peft:
        results["custom_eval_files"]["mid"] = os.path.join(
            evals_dir, f"custom_eval_{eval_type}_mid.csv"
        )
    
    Utilities.save_experiment_results(args.output, args.exp_id, results)
    Utilities.update_experiment_metadata(args.output, args.exp_id, results)


if __name__ == "__main__":
    main()
