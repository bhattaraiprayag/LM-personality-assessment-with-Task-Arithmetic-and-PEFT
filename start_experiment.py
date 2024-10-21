# start_experiment.py

from src.utilities import Utilities
from src.data_manager import DataManager
from src.model_manager import CLMModel
from src.eval_manager import EvalManager

Utilities.suppress_warnings()

import warnings
import os
import torch
import math
import pytorch_lightning as pl
from argparse import Namespace

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def main():
    exp_utils = Utilities()
    args, loggers, callbacks, device_config = exp_utils.housekeep()
    data_manager = DataManager(args)
    data_manager.setup(args)
    tokenizer = data_manager.tokenizer
    len_train_loader = len(data_manager.train_dataloader())
    total_steps = math.ceil(len_train_loader / args.grad_steps) * args.epochs if device_config['multi_gpu'] == False else math.ceil((len_train_loader * args.epochs * len(device_config['devices'])) / args.grad_steps)
    warmup_steps = max(1, int(total_steps * min(max(args.warmup_ratio, 0.03), 0.1)))

    model_hparams = Namespace(
        lr=args.lr,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        accumulate_grad_batches=int(args.grad_steps)
    )
    model = CLMModel(
        args.model_name, model_hparams, args.use_peft, args.scale_peft, tokenizer
    )

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
        "has an active imagination."
    ]

    model.eval()
    with torch.no_grad():
        personality_eval_pre = EvalManager.extract_answers(model, tokenizer, sample_question, possible_answers, temps=temperatures)

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
        val_dataloaders=data_manager.val_dataloader()
    )

    results = {}
    train_metrics = trainer.callback_metrics
    results['train_metrics'] = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in train_metrics.items()}

    test_results = trainer.test(
        model,
        dataloaders=data_manager.test_dataloader()
    )
    if test_results:
        test_metrics = test_results[0]
        results['test_metrics'] = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_metrics.items()}

    model.eval()
    with torch.no_grad():
        personality_eval_post = EvalManager.extract_answers(model, tokenizer, sample_question, possible_answers, temps=temperatures)
    
    personality_eval_pre_dict = personality_eval_pre.to_dict(orient='records')
    personality_eval_post_dict = personality_eval_post.to_dict(orient='records')
    results['personality_eval_pre'] = personality_eval_pre_dict
    results['personality_eval_post'] = personality_eval_post_dict
    exp_utils.save_experiment_results(args.output, args.exp_id, results)
    exp_utils.update_experiment_metadata(args.base_output_dir, args.exp_id, results)


if __name__ == "__main__":
    main()