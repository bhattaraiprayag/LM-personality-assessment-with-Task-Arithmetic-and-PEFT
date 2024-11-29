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
        # device = next(model.parameters()).device
        model = model.to(device)
        model.eval()

        # input_text = [
        #     f"{tokenizer.bos_token}{instr_tag_before}{question}{instr_tag_after} {answer}{tokenizer.eos_token}"
        #     for answer in answers
        # ]
        input_text = [
            f"{tokenizer.bos_token}{instr_tag_before}{question}{instr_tag_after}{tokenizer.eos_token}"
        ]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(device)

        input_length = input_ids.shape[1]
        result_df = pd.DataFrame()

        with torch.no_grad():
            for temp in temps:
                # Initialize lists to store results
                answer_texts = []
                answer_probs = []
                answer_values = []

                # For each possible answer, compute the probability that the model generates that answer
                for answer, value in zip(answers, values):
                    # Tokenize the answer
                    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

                    # Initialize probability for this answer
                    prob = 1.0

                    # Initialize the context with the input_ids (question)
                    context_ids = input_ids.clone()

                    # Compute probability of each token in the answer
                    for token_id in answer_ids:
                        # Get model outputs for the current context
                        outputs = model(input_ids=context_ids, attention_mask=context_ids.ne(tokenizer.pad_token_id).long())
                        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

                        # Get the logits for the last token
                        last_token_logits = logits[0, -1, :] / temp  # Apply temperature

                        # Compute probabilities
                        token_probs = torch.softmax(last_token_logits, dim=-1)

                        # Get the probability of the next token
                        token_prob = token_probs[token_id].item()

                        # Multiply the probabilities
                        prob *= token_prob

                        # Append the token_id to context_ids for the next iteration
                        next_token_id = torch.tensor([[token_id]], device=device)
                        context_ids = torch.cat([context_ids, next_token_id], dim=1)

                    # Store the results
                    answer_texts.append(answer)
                    answer_probs.append(prob)
                    answer_values.append(value)

                # # Normalize the probabilities
                # total_prob = sum(answer_probs)
                # norm_probs = [p / total_prob for p in answer_probs]

                # Create DataFrame for this temperature
                temp_df = pd.DataFrame({
                    'answer': answer_texts,
                    # 'value': answer_values,
                    'prob': answer_probs,
                    # 'norm_probs': norm_probs,
                    'temp': temp
                })

                result_df = pd.concat([result_df, temp_df], ignore_index=True)

        return result_df.reset_index(drop=True)

        # # input_ids, attention_mask = tokenizer(
        # #     input_texts, padding=True, return_tensors="pt"
        # # ).values()
        # input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)
        # attention_mask[:, 0] = 0
        # question_length = len(tokenizer(question)["input_ids"])
        # model = model.to(device)
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # result_df = pd.DataFrame()
        # for temp in temps:
        #     if variant in (1, 2):
        #         probs = torch.log_softmax(outputs.logits / temp, dim=-1).detach()
        #         _ids = input_ids[:, 1:]
        #         gen_probs = torch.gather(probs, 2, _ids[:, :, None]).squeeze(-1)
        #         if variant == 2:
        #             gen_probs = torch.log_softmax(gen_probs / temp, dim=0)
        #         batch_df = pd.DataFrame()
        #         for input_sentence, input_probs, answer, value in zip(
        #             _ids, gen_probs, answers, values
        #         ):
        #             text_sequence = []
        #             for token, p in list(zip(input_sentence, input_probs))[
        #                 question_length:
        #             ]:
        #                 if token not in tokenizer.all_special_ids and token != 29871:
        #                     text_sequence.append((tokenizer.decode(token), p.item()))
        #             answer_df = pd.DataFrame(text_sequence, columns=["token", "prob"])
        #             answer_df["answer"] = answer
        #             answer_df["value"] = value
        #             batch_df = pd.concat([batch_df, answer_df])
        #         batch_df = (
        #             batch_df.drop(columns=["token"])
        #             .groupby(["answer", "value"])
        #             .mean()
        #             .reset_index()
        #         )
        #         batch_df["temp"] = temp
        #         batch_df["answer"] = batch_df["answer"].replace("</s>", "N/A")
        #         result_df = pd.concat([result_df, batch_df])
        # one_word_prompt = (
        #     "Complete the following with a one-word adjective that aptly describes "
        #     "your personality: "
        # )
        # combined_one_word_question = one_word_prompt + question + " is"
        # question_input_ids = tokenizer(
        #     f"{tokenizer.pad_token}{combined_one_word_question}", return_tensors="pt"
        # ).input_ids.to(device)
        # attention_mask = question_input_ids.ne(tokenizer.pad_token_id).long()
        # generated_outputs = model.model.generate(
        #     question_input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=1,
        #     num_return_sequences=1,
        #     do_sample=False,  # Greedy decoding for deterministic output, if False
        #     pad_token_id=tokenizer.pad_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        #     temperature=0.5,  # Sampling temperature
        #     # top_p=0.95,
        # )
        # decoded_answer = tokenizer.decode(
        #     generated_outputs[0], skip_special_tokens=True
        # ).strip()
        # # print(f"Model's continuation of the question: {decoded_answer}")

        # # Normalize the probabilities (for variant 1)
        # if variant == 1:
        #     log_probs = np.array(result_df["prob"].values, dtype=np.float64)
        #     log_probs_shifted = log_probs - np.max(log_probs)
        #     probs = np.exp(log_probs_shifted)
        #     normalized_probs = probs / probs.sum()
        #     result_df["norm_probs"] = normalized_probs
        # return result_df.reset_index(drop=True)
