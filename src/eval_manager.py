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
    def extract_answers(model, tokenizer, question: str,
                        answers: list, values: Optional[list] = None,
                        temps: Optional[List[float]] = None, variant=1,
                        instr_tag_before="", instr_tag_after=""
                        ) -> pd.DataFrame:
        """
        Generates model responses to a set of questions and computes
        probabilities for each answer.

        Args:
            model: The language model to evaluate.
            tokenizer: Tokenizer corresponding to the model.
            question (str): The question prompt to evaluate.
            answers (list): List of possible answers to the question.
            values (Optional[list]): List of values associated with
                each answer.
            temps (Optional[List[float]]): List of temperatures for
                sampling.
            variant (int): Variant of the evaluation method to use.
            instr_tag_before (str): Instruction tag to prepend before
                the question.
            instr_tag_after (str): Instruction tag to append after the
                question.

        Returns:
            pd.DataFrame: DataFrame containing probabilities and other
                details for each answer.
        """
        tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
        tokenizer.pad_token = tokenizer.eos_token
        if values is None:
            values = [1] * len(answers)
        if temps is None:
            temps = [1.0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        input_text = [
            f"{tokenizer.bos_token}{instr_tag_before}{question}{instr_tag_after}{tokenizer.eos_token}"
        ]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)
        input_length = input_ids.shape[1]
        result_df = pd.DataFrame()

        with torch.no_grad():
            for temp in temps:
                answer_texts = []
                answer_probs = []
                answer_values = []
                for answer, value in zip(answers, values):
                    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                    prob = 1.0
                    context_ids = input_ids.clone()
                    for token_id in answer_ids:
                        outputs = model(input_ids=context_ids, attention_mask=context_ids.ne(tokenizer.pad_token_id).long())
                        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
                        last_token_logits = logits[0, -1, :] / temp  # Apply temperature
                        token_probs = torch.softmax(last_token_logits, dim=-1)
                        token_prob = token_probs[token_id].item()
                        prob *= token_prob
                        next_token_id = torch.tensor([[token_id]], device=device)
                        context_ids = torch.cat([context_ids, next_token_id], dim=1)
                    answer_texts.append(answer)
                    answer_probs.append(prob)
                    answer_values.append(value)

                # total_prob = sum(answer_probs)
                # norm_probs = [p / total_prob for p in answer_probs]

                temp_df = pd.DataFrame({
                    'answer': answer_texts,
                    # 'value': answer_values,
                    'prob': answer_probs,
                    # 'norm_probs': norm_probs,
                    'temp': temp
                })
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
        return result_df.reset_index(drop=True)
