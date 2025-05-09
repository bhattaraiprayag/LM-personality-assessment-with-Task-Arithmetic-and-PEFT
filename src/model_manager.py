# src/model_manager.py
"""
Module defining the CLMModel class for language model training
using PyTorch Lightning.
"""
import pytorch_lightning as pl
import torch

from argparse import Namespace
from typing import Dict, List
from torch import nn
from transformers import AdamW
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class CLMModel(pl.LightningModule):
    """
    PyTorch Lightning module for training causal language models.

    Attributes:
        model: The pre-trained language model.
        model_hparams (Namespace): Hyperparameters for training.
        loss_fn (nn.CrossEntropyLoss): Loss function for training.
        metrics (Dict[str, List[float]]): Dictionary to store training metrics.
    """

    def __init__(self, model, model_hparams: Namespace):
        """
        Initializes the CLMModel with the given model and hyperparameters.

        Args:
            model: The pre-trained language model to train.
            model_hparams (Namespace): Hyperparameters for the model.
        """
        super().__init__()
        self.model = model
        self.model_hparams = model_hparams
        self.save_hyperparameters(self.model_hparams)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=1e-7)
        self.metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "val_perplexity": [],
            "test_perplexity": [],
        }
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.test_step_outputs: List[Dict[str, torch.Tensor]] = []

    def forward(self, *args, **kwargs) -> CausalLMOutputWithCrossAttentions:
        """
        Forward pass of the model.

        Returns:
            Model output from the forward pass.
        """
        return self.model(*args, **kwargs)

    def compute_loss(
        self,
        outputs: CausalLMOutputWithCrossAttentions,
        _batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss between model outputs and targets.

        Args:
            outputs (CausalLMOutputWithCrossAttentions): Model outputs.
            batch (Dict[str, torch.Tensor]): Batch of input data.

        Returns:
            torch.Tensor: Computed loss.
        """
        return outputs.loss

    def training_step(self, batch: dict, _batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a single training step.

        Args:
            batch (dict): Batch of input data.
            _batch_idx (int): Batch index (unused).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.
        """
        outputs = self(**batch)
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def validation_step(self, batch: dict, _batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a single validation step.

        Args:
            batch (dict): Batch of input data.
            _batch_idx (int): Batch index (unused).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.
        """
        outputs = self(**batch)
        loss = outputs.loss
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.validation_step_outputs.append({"loss": loss})
        return {"loss": loss}

    def test_step(self, batch: dict, _batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Performs a single test step.

        Args:
            batch (dict): Batch of input data.
            _batch_idx (int): Batch index (unused).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.
        """
        outputs = self(**batch)
        loss = outputs.loss
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.test_step_outputs.append({"loss": loss})
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        """
        Calculates and logs average validation loss and perplexity at the
        end of each validation epoch.
        """
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        perplexity = torch.exp(avg_loss)
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_perplexity", perplexity, prog_bar=True)
        self.metrics["val_loss"].append(avg_loss.item())
        self.metrics["val_perplexity"].append(perplexity.item())
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """
        Calculates and logs average test loss and perplexity at the
        end of testing.
        """
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()
        perplexity = torch.exp(avg_loss)
        self.log("test_loss", avg_loss, prog_bar=True)
        self.log("test_perplexity", perplexity, prog_bar=True)
        self.metrics["test_loss"].append(avg_loss.item())
        self.metrics["test_perplexity"].append(perplexity.item())
        self.test_step_outputs.clear()

    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Retrieves the stored training metrics.

        Returns:
            Dict[str, List[float]]: Dictionary of metrics.
        """
        return self.metrics

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
            Optimizer: The optimizer to use.
        """
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.model_hparams.lr
        )
        return optimizer
