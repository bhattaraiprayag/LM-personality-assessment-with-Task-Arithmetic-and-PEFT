# src/eval_manager.py
"""
Module for evaluating language models by extracting and analyzing
their responses to specific prompts.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import torch


class EvalManager:
    """
    Class containing methods to evaluate a language model's responses
    to personality assessment questions.
    """

    @staticmethod
    def extract_answers(model,
                        tok,
                        question: str,
                        answers: List[str],
                        temps: List[float]) -> pd.DataFrame:
        """
        Vectorised, numerically-stable version that returns one row
        per (temperature, answer) pair with a proper probability.
        """
        device = next(model.parameters()).device

        # --- build the batch -------------------------------------------------
        bos = tok.bos_token or ""
        question_ids = tok.encode(bos + question, add_special_tokens=False)
        q_len = len(question_ids)

        prompts = [bos + question + ans for ans in answers]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)  # [B, L]

        with torch.no_grad():
            logits = model(**enc).logits                                  # [B, L, V]

        # --- token-log-probs of the actually generated tokens --------------
        logp_token = torch.log_softmax(logits, dim=-1)\
                        .gather(2, enc.input_ids.unsqueeze(-1))\
                        .squeeze(-1)                                       # [B, L]

        # --- build a mask that selects *only* the answer tokens ------------
        seq_len = enc.input_ids.size(1)
        idx = torch.arange(seq_len, device=device).expand(len(answers), -1)

        ans_lens = torch.tensor(
            [len(tok.encode(a, add_special_tokens=False)) for a in answers],
            device=device).unsqueeze(1)

        mask = (idx >= q_len) & (idx < q_len + ans_lens)                  # [B, L]

        seq_logp = (logp_token * mask).sum(1)                             # [B]

        # --- temperature scaling + stable softmax --------------------------
        rows = []
        for T in temps:
            if T == 0:
                winner = torch.argmax(seq_logp)
                probs = torch.zeros_like(seq_logp)
                probs[winner] = 1.0
            else:
                probs = torch.softmax(seq_logp / T, dim=0)                    # [B]
            # probs = torch.softmax(seq_logp / T, dim=0)                    # [B]
            for ans, p in zip(answers, probs.tolist()):
                rows.append({"temp": T, "answer": ans, "prob": p})
            # print(f"T={T}, sum={probs.sum().item()}")

        return pd.DataFrame(rows)
