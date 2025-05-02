import os
from typing import Any, Dict, List

import torch
from peft import get_peft_model_state_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.utils.main import Utilities
from src.utils.pipeline import perform_evaluation, perform_ipip120_evaluation
from src.eval_results_manager import EvalResultsManager


class MidEpochCheckpointCallback(Callback):
    """Custom callback for saving LoRA checkpoints and performing mid-epoch evaluation."""
    
    def __init__(
        self,
        args: Any,
        tokenizer: Any,
        sample_question: str,
        possible_answers: List[str],
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
            sample_question: Question template for evaluation
            possible_answers: Possible answers for evaluation
            temperatures: List of temperatures for evaluation
            peft_scales: List of PEFT scales for evaluation
            total_steps: Total number of training steps
            eval_type: Type of evaluation (personality, emotion)
        """
        self.args = args
        self.tokenizer = tokenizer
        self.sample_question = sample_question
        self.possible_answers = possible_answers
        self.temperatures = temperatures
        self.peft_scales = peft_scales
        self.eval_type = eval_type
        
        # Calculate save intervals (N/5 steps)
        self.save_intervals = [i * (total_steps // 5) for i in range(1, 6)]
        self.saved_steps = set()
        
        # Create directories
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
        
        # Check if we should save at this step
        if current_step in self.save_intervals and current_step not in self.saved_steps:
            self.saved_steps.add(current_step)
            
            # Save LoRA weights
            self._save_lora_weights(pl_module, current_epoch, current_step)
            
            # Perform evaluation
            self._perform_mid_epoch_eval(pl_module, current_epoch, current_step)
    
    def _save_lora_weights(
        self, pl_module: pl.LightningModule, epoch: int, step: int
    ) -> None:
        """Save LoRA adapter weights."""
        # Get the PEFT model from the Lightning module
        peft_model = pl_module.model
        
        # Get LoRA state dict
        lora_state_dict = get_peft_model_state_dict(peft_model)
        
        # Create filename
        filename = f"epoch{epoch:02d}_step{step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save weights
        torch.save(lora_state_dict, filepath)

    def _perform_mid_epoch_eval(
        self, pl_module: pl.LightningModule, epoch: int, step: int
    ) -> None:
        """Perform mid-epoch evaluation."""
        # Set model to eval mode
        pl_module.eval()

        # Container for all scale results
        all_results = {}

        # Perform evaluation for each scale
        for scale in self.peft_scales:
            # Run evaluation
            eval_results = perform_evaluation(
                pl_module.model,  # Pass the PEFT model directly
                self.tokenizer,
                self.temperatures,
                self.sample_question,
                self.possible_answers,
                self.args,
                scale
            )
            
            # Store results for this scale
            all_results[f"scale_{scale}"] = eval_results.to_dict(orient="records")
            
            # Perform IPIP-120 evaluation for pandora dataset
            if self.args.dataset == "pandora":
                ipip_results = perform_ipip120_evaluation(
                    pl_module.model,
                    self.tokenizer,
                    self.args,
                    scale
                )
                
                # Save IPIP-120 results with epoch and step information
                EvalResultsManager.save_ipip120_eval_results(
                    output_dir=self.args.output,
                    experiment_id=self.args.exp_id,
                    phase="mid",
                    results=ipip_results,
                    scale=scale,
                    epoch=epoch,
                    step=step
                )
        
        # Save results using the EvalResultsManager
        EvalResultsManager.save_custom_eval_results(
            output_dir=self.args.output,
            experiment_id=self.args.exp_id,
            phase="mid",
            eval_type=self.eval_type,
            question=self.sample_question,
            answers=self.possible_answers,
            results=all_results,
            epoch=epoch,
            step=step
        )
        
        # Set model back to train mode
        pl_module.train() 
