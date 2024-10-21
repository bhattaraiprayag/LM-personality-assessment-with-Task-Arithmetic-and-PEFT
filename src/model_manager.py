# src/model_manager.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AdamW, AutoTokenizer
from argparse import Namespace
from typing import Optional
from src.peft_manager import PEFTManager

class CLMModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_hparams: Namespace,
        use_peft: Optional[str] = None,
        scale_peft: Optional[float] = 1.0,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.model_hparams = model_hparams
        self.save_hyperparameters(self.model_hparams)
        self.use_peft = use_peft
        self.scale_peft = scale_peft
        self.tokenizer = tokenizer
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "val_perplexity": [],
            "test_perplexity": []
        }
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        if self.use_peft:
            self.model = PEFTManager.apply_peft(self.model, self.use_peft, self.scale_peft)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=1e-7)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        return outputs.loss

    def training_step(self, batch: dict, batch_idx: int):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch: dict, batch_idx: int):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.append({'loss': loss})
        return {'loss': loss}

    def test_step(self, batch: dict, batch_idx: int):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.test_step_outputs.append({'loss': loss})
        return {'loss': loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        perplexity = torch.exp(avg_loss)
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)
        self.metrics["val_loss"].append(avg_loss.item())
        self.metrics["val_perplexity"].append(perplexity.item())
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        perplexity = torch.exp(avg_loss)
        self.log('test_loss', avg_loss, prog_bar=True)
        self.log('test_perplexity', perplexity, prog_bar=True)
        self.metrics["test_loss"].append(avg_loss.item())
        self.metrics["test_perplexity"].append(perplexity.item())
        self.test_step_outputs.clear()

    def get_metrics(self):
        return self.metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.model_hparams.lr)
        return optimizer