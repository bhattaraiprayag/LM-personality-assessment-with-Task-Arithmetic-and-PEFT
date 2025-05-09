# src/eval_manager.py
"""
Module for evaluating language models by extracting and analyzing
their responses to specific prompts.
"""
from __future__ import annotations  # CAUTION: has to always be placed at the top of the script

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from typing import List, Optional, Dict, Union
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_inventory(name: str) -> Dict:
    """
    Returns a dict with keys:
        • anchors  (List[str])
        • items    (List[str])
        • question_stem
        • statement_template  (must contain '{item}')
    """
    from experiment_config import INVENTORIES
    import pandas as pd

    cfg = INVENTORIES[name].copy()
    if "items" not in cfg:  # CSV variant (IPIP‑120): read once, cache in cfg
        df = pd.read_csv(cfg.pop("items_file"))
        cfg["items"] = df["phrase"].tolist()
    return cfg


class EvalManager:
    """
    Class containing methods to evaluate a language model's responses
    to personality assessment questions.
    """
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda",
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self.device = torch.device(device)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()

        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    @torch.no_grad()
    def score_likert(
        self,
        items: List[str] | None = None,
        anchors: List[str] | None = None,
        *,
        question_stem: str | None = None,
        statement_template: str = "STATEMENT: {item}",
        inventory_name: str | None = None,
        include_options: Union[bool, str] = "both",
        batch_size: int = 5,
        return_dataframe: bool = True,
    ):
        """
        Supply **either** (`items`, `anchors`, `question_stem`) yourself
        **or** just `inventory_name="BFI10" | "PANASX" | "IPIP120" | …`.
        """
        import pandas as pd
        if inventory_name is not None:
            inv = get_inventory(inventory_name)
            items = inv["items"]
            anchors = inv["anchors"]
            question_stem = inv["question_stem"]
            statement_template = inv["statement_template"]
        if None in (items, anchors, question_stem):
            raise ValueError(
                "Must provide `inventory_name` or explicit `items`, `anchors`, `question_stem`."
            )
        items = [item.lower() for item in items]

        if include_options == "both":
            variants = (True, False)
        elif include_options == "include":
            variants = (True,)
        elif include_options == "exclude":
            variants = (False,)
        else:
            raise ValueError(f"Invalid value for `include_options`: {include_options!r}")

        anchor_ids = [
            self.tokenizer(" " + a, add_special_tokens=False).input_ids for a in anchors
        ]
        anchor_lens = [len(ids) for ids in anchor_ids]

        all_variant_probs = []
        variant_tags = []
        dl = DataLoader(items, batch_size=batch_size, shuffle=False)

        for show_options in variants:
            variant_probs = []
            for batch_items in dl:
                toks, mask = self._collate_batch(
                    batch_items=batch_items,
                    anchor_ids=anchor_ids,
                    anchor_lens=anchor_lens,
                    question_stem=question_stem,
                    statement_template=statement_template,
                    anchors=anchors,
                    include_options=show_options,
                )
                avg_loss = self._forward_avg_loss(toks, mask)
                probs = self._avg_loss_to_probs(avg_loss, len(batch_items))
                variant_probs.append(probs)

            variant_tensor = torch.cat(variant_probs, dim=0)
            all_variant_probs.append(variant_tensor)
            tag = "include" if show_options else "exclude"
            variant_tags.extend([tag] * len(items))

        all_probs = torch.vstack(all_variant_probs)

        if not return_dataframe:
            return all_probs

        num_cols = list(range(1, len(anchors) + 1))
        df = pd.DataFrame(all_probs.cpu().numpy(), columns=num_cols)
        df.insert(0, "item", items * len(variants))
        df.insert(1, "likert_in_prompt", variant_tags)
        return df

    def _build_prompt(
        self,
        item: str,
        anchors: List[str],
        question_stem: str,
        statement_template: str,
        include_options: bool,
    ) -> str:
        if include_options:
            bullets = "\n".join(f"{a}" for a in anchors)
            options_block = f"OPTIONS (choose one):\n{bullets}\n\n"
        else:
            options_block = ""

        statement = statement_template.format(item=item)

        return (
            f"QUESTION: {question_stem}\n\n"
            f"{statement}\n\n"
            f"{options_block}"
            f"YOUR RESPONSE: "
        )

    def _collate_batch(
        self,
        batch_items: List[str],
        anchor_ids: List[List[int]],
        anchor_lens: List[int],
        question_stem: str,
        statement_template: str,
        anchors: List[str],
        *,
        include_options: bool,
    ):
        prompts = [
            self._build_prompt(
                item, anchors, question_stem, statement_template, include_options
            )
            for item in batch_items
        ]
        prompt_ids = [self.tokenizer(p).input_ids for p in prompts]
        ctx_lens = [len(ids) for ids in prompt_ids]

        tok_rows, mask_rows = [], []
        for ctx, ctx_len in zip(prompt_ids, ctx_lens):
            for a_ids, a_len in zip(anchor_ids, anchor_lens):
                row_toks = ctx + a_ids
                row_mask = [0] * ctx_len + [1] * a_len
                tok_rows.append(row_toks)
                mask_rows.append(row_mask)

        max_len = max(len(r) for r in tok_rows)
        toks = torch.zeros(len(tok_rows), max_len, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(toks)

        for i, (row_toks, row_mask) in enumerate(zip(tok_rows, mask_rows)):
            toks[i, : len(row_toks)] = torch.tensor(row_toks, device=self.device)
            mask[i, : len(row_mask)] = torch.tensor(row_mask, device=self.device)

        return toks, mask

    def _forward_avg_loss(self, toks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.model(toks).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_toks = toks[..., 1:].contiguous()

        loss_flat = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_toks.view(-1),
            reduction="none",
        ).view(toks.size(0), -1)

        shift_mask = mask[..., 1:]
        sum_loss = (loss_flat * shift_mask).sum(dim=1)
        token_count = shift_mask.sum(dim=1)
        return sum_loss / token_count

    def _avg_loss_to_probs(self, avg_loss: torch.Tensor, batch_items: int) -> torch.Tensor:
        loss = avg_loss.view(batch_items, -1)
        return torch.softmax(-loss, dim=1)
