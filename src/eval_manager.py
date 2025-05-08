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

    @staticmethod
    def extract_ipip120(model,
                       tok,
                       ipip_csv_path: str,
                       instructions: str,
                       likert_options: List[str],
                       likert_values: List[int]) -> pd.DataFrame:
        """
        Evaluate model on IPIP-120 personality inventory.
        
        Args:
            model: The language model to evaluate
            tok: Tokenizer for the model
            ipip_csv_path: Path to CSV file with IPIP items
            instructions: Instructions for the IPIP assessment
            likert_options: List of Likert scale options
            likert_values: Corresponding numeric values for Likert options
            
        Returns:
            pd.DataFrame: DataFrame with IPIP evaluation results
        """
        import pandas as pd
        from tqdm import tqdm
        import numpy as np
        
        # Read IPIP items
        df = pd.read_csv(ipip_csv_path)
        device = next(model.parameters()).device
        
        # Ensure pad token is set
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        
        model.eval()
        batch_size = 30  # Process items in batches for efficiency
        all_results = []
        
        # Process IPIP items in batches
        for batch_start in tqdm(range(0, len(df), batch_size)):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            # Prepare prompts for all items in the batch and all likert options
            all_prompts = []
            item_indices = []
            option_indices = []
            
            # Create a matrix of prompts: items x options
            for item_idx, (_, row) in enumerate(batch_df.iterrows()):
                phrase = row["phrase"]
                phrase = phrase.lower()
                context = f"{instructions}{phrase}\n\nYour likert answer choice:"
                
                for option_idx, option in enumerate(likert_options):
                    all_prompts.append(context + " " + option)
                    item_indices.append(item_idx)
                    option_indices.append(option_idx)
            
            # Convert item_indices and option_indices to tensors for later use
            item_indices = torch.tensor(item_indices, device=device)
            option_indices = torch.tensor(option_indices, device=device)
            
            # Tokenize all prompts at once
            bos = tok.bos_token or ""
            all_prompt_texts = [bos + prompt for prompt in all_prompts]
            
            # Get context lengths for each prompt to mask out
            context_lengths = []
            option_lengths = []
            
            for i, prompt in enumerate(all_prompts):
                option_idx = option_indices[i].item()
                option = likert_options[option_idx]
                
                # Calculate context length (everything except the option)
                full_text = bos + prompt
                option_text = " " + option
                context_len = len(tok.encode(full_text, add_special_tokens=False)) - len(tok.encode(option_text, add_special_tokens=False))
                context_lengths.append(context_len)
                option_lengths.append(len(tok.encode(option_text, add_special_tokens=False)))
            
            # # Batch encode all prompts
            # encodings = tok(all_prompt_texts, return_tensors="pt", padding=True).to(device)
            
            # # Get model outputs for the whole batch
            # with torch.no_grad():
            #     outputs = model(**encodings)
            #     logits = outputs.logits
            MICRO = 16
            option_scores = torch.empty(len(all_prompts), device=device)

            for mb_start in range(0, len(all_prompts), MICRO):
                mb_end = min(mb_start + MICRO, len(all_prompts))
                encodings = tok(
                    all_prompt_texts[mb_start:mb_end],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.no_grad(), torch.cuda.amp.autocast():
                    logits = model(**encodings).logits
                
                # Calculate log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, encodings.input_ids.unsqueeze(-1)).squeeze(-1)
                
                mb_ctx = context_lengths[mb_start:mb_end]
                mb_olen = option_lengths[mb_start:mb_end]
                mb_masks = torch.zeros_like(encodings.input_ids, dtype=torch.bool)

                for j, ctx_len in enumerate(mb_ctx):
                    mb_masks[j, ctx_len:ctx_len + mb_olen[j]] = True

                masked = token_log_probs * mb_masks
                sums = masked.sum(1)
                counts = mb_masks.sum(1)
                mb_scores = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
                option_scores[mb_start:mb_end] = mb_scores

                # # Create masks to select only the option tokens
                # seq_len = encodings.input_ids.size(1)
                # position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(len(all_prompts), -1)
                
                # # Create a mask for each prompt that selects only the option tokens
                # option_masks = torch.zeros_like(encodings.input_ids, dtype=torch.bool)
                # for i, ctx_len in enumerate(context_lengths):
                #     option_masks[i, ctx_len:ctx_len + option_lengths[i]] = True
                
                # # Apply the masks to get log probs for only the option tokens
                # masked_log_probs = token_log_probs * option_masks
                
                # # Calculate mean log prob for each option (where mask is True)
                # # option_scores = torch.zeros(len(all_prompts), device=device)
                # # for i in range(len(all_prompts)):
                # for i in range(mb_start, mb_end):
                #     if option_masks[i].sum() > 0:  # Avoid division by zero
                #         option_scores[i] = masked_log_probs[i].sum() / option_masks[i].sum()

                del logits, log_probs, token_log_probs, masked, mb_masks
                torch.cuda.empty_cache()
            
            # Reshape scores to [num_items, num_options]
            num_items_in_batch = batch_end - batch_start
            scores_matrix = torch.zeros(num_items_in_batch, len(likert_options), device=device)
            
            for i in range(len(all_prompts)):
                item_idx = item_indices[i].item()
                option_idx = option_indices[i].item()
                scores_matrix[item_idx, option_idx] = option_scores[i]
            
            # Find best option for each item
            best_indices = torch.argmax(scores_matrix, dim=1)
            
            # Process results for this batch
            for i, (_, row) in enumerate(batch_df.iterrows()):
                q_id, phrase, reverse = row["q_id"], row["phrase"], row["reverse_code"]
                
                best_index = best_indices[i].item()
                assigned_score = likert_values[best_index]
                
                # Reverse score if needed
                if reverse:
                    assigned_score = 6 - assigned_score
                
                # Store result
                result_row = {
                    "q_id": q_id,
                    "phrase": phrase,
                    "reverse_code": reverse,
                    "assigned_score": assigned_score,
                    "max_likert_option": likert_options[best_index],
                    "max_logp": scores_matrix[i, best_index].item()
                }
                all_results.append(result_row)
        
        return pd.DataFrame(all_results)
