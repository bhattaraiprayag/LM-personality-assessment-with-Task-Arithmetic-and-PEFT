# start_experiment.py
"""
Script to initialize and run an experiment, including data preparation, model
training, and evaluation.
"""
import os
import pytorch_lightning as pl
import torch

from peft import get_peft_model_state_dict
from tqdm import tqdm

import experiment_config
from src.utils.main import Utilities
from src.utils.pipeline import (calculate_warmup_steps, initialize_model,
                               run_inventory_eval, prepare_data,
                               setup_experiment)
from src.utils.callbacks import MidEpochCheckpointCallback
from src.utils.perplexity import run_perplexity_benchmark

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

    # Define evaluation parameters based on dataset
    peft_scales = experiment_config.peft_scales
    temperatures = experiment_config.temperatures

    # Calculate total steps for mid-epoch checkpoints
    total_steps = len(data_manager.train_dataloader()) * args.epochs // int(args.grad_steps)

    # Add mid-epoch checkpoint callback if using PEFT
    if args.use_peft:
        mid_epoch_callback = MidEpochCheckpointCallback(
            args=args,
            tokenizer=tokenizer,
            temperatures=temperatures,
            peft_scales=peft_scales,
            total_steps=total_steps,
            eval_type=args.dataset  # "emotion" or "pandora"
        )
        callbacks.append(mid_epoch_callback)

    # Log Train and Test metrics
    results = {}

    # Finetune
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
        # val_check_interval=0.5,   # Uncomment for validation twice per epoch
        precision=16,
        deterministic=True,
    )
    trainer.fit(model, train_dataloaders=data_manager.train_dataloader(),
                val_dataloaders=data_manager.val_dataloader())
    train_metrics = trainer.callback_metrics

    # Test
    model.eval()
    test_results = trainer.test(model, dataloaders=data_manager.test_dataloader())

    # Psychometric Evaluation (Post-Finetuning)
    torch.cuda.empty_cache()
    if args.use_peft:
        for scale in tqdm(peft_scales, desc="POST Eval across LoRA scales..."):
            run_inventory_eval(model, tokenizer, args, phase="post", scale=scale)
    else:
        run_inventory_eval(model, tokenizer, args, phase="post")

    # Benchmark Perplexity
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    benchmark_results = run_perplexity_benchmark(model.model, tokenizer, args.seed)
    print(f"Perplexity: {benchmark_results}")

    # Access the wrapped model and save LoRA weights or full model
    if args.use_peft:
        lora_state_dict = get_peft_model_state_dict(model.model)
        torch.save(lora_state_dict, os.path.join(args.exp_out_dir, "lora_final.pt"))
    else:
        model.model.save_pretrained(args.exp_out_dir)

    # Compile & Save Train and Test metrics
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
    results["benchmark_perplexity"] = benchmark_results
    evals_dir = os.path.join(args.output, args.exp_id, "evals")
    results["custom_eval_files"] = {
        "post": os.path.join(evals_dir, "post_epoch_results.csv"),
    }    
    if args.use_peft:
        results["custom_eval_files"]["mid"] = os.path.join(
            evals_dir, "mid_epoch_results.csv"
        )
    Utilities.save_experiment_results(args.output, args.exp_id, results)
    Utilities.update_experiment_metadata(args.output, args.exp_id, results)


if __name__ == "__main__":
    main()
