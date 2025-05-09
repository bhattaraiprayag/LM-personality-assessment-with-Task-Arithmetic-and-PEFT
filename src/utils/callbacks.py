# src/utils/callbacks.py
"""
Custom callback for saving LoRA checkpoints and performing mid-epoch evaluation.
"""
import os
import torch
import pytorch_lightning as pl

from typing import Any, Dict, List
from tqdm import tqdm
from peft import get_peft_model_state_dict
from pytorch_lightning.callbacks import Callback

from src.utils.main import Utilities
from src.utils.pipeline import run_inventory_eval
from src.eval_results_manager import EvalResultsManager


class MidEpochCheckpointCallback(Callback):
    """Custom callback for saving LoRA checkpoints and performing mid-epoch evaluation."""
    def __init__(
        self,
        args: Any,
        tokenizer: Any,
        temperatures: List[float],
        peft_scales: List[float],
        total_steps: int,
        eval_type: str,
    ):
        """
        Initialize the callback.
        
        Args:
            args: Experiment arguments
            tokenizer: Tokenizer for evaluation
            temperatures: List of temperatures for evaluation
            peft_scales: List of PEFT scales for evaluation
            total_steps: Total number of training steps
            eval_type: Type of evaluation (personality, emotion)
        """
        self.args = args
        self.tokenizer = tokenizer
        self.temperatures = temperatures
        self.peft_scales = peft_scales
        self.eval_type = eval_type

        self.save_intervals = [i * (total_steps // 5) for i in range(1, 6)]
        self.saved_steps = set()

        self.checkpoint_dir = os.path.join(args.exp_out_dir, "checkpoints")
        self.eval_dir = os.path.join(args.exp_out_dir, "evals")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called at the end of each training batch."""
        if not self.args.use_peft:
            return

        current_step = trainer.global_step
        current_epoch = trainer.current_epoch

        if current_step in self.save_intervals and current_step not in self.saved_steps:
            self.saved_steps.add(current_step)            
            self._save_lora_weights(pl_module, current_epoch, current_step)
            self._perform_mid_epoch_eval(pl_module, current_epoch, current_step)
    
    def _save_lora_weights(
        self, pl_module: pl.LightningModule, epoch: int, step: int
    ) -> None:
        """Save LoRA adapter weights."""
        peft_model = pl_module.model
        lora_state_dict = get_peft_model_state_dict(peft_model)
        filename = f"epoch{epoch:02d}_step{step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(lora_state_dict, filepath)

    def _perform_mid_epoch_eval(
        self, pl_module: pl.LightningModule, epoch: int, step: int
    ) -> None:
        """Perform mid-epoch evaluation."""
        pl_module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        pl_module.eval()
        orig_device = next(pl_module.parameters()).device
        pl_module.to("cpu")

        with torch.no_grad():
            for scale in tqdm(self.peft_scales, desc="MID Eval across LoRA scales..."):
                run_inventory_eval(
                    pl_module.model,
                    self.tokenizer,
                    self.args,
                    phase="mid",
                    epoch=epoch,
                    step=step,
                    scale=scale,
                )
        pl_module.to(orig_device)
        torch.cuda.empty_cache()
        pl_module.train()
