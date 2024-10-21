# src/eval_manager.py

import torch
import pandas as pd
import numpy as np


class EvalManager:
    @staticmethod
    def extract_answers(model, tokenizer, question: str, answers: list, values: list = None, temps: list = [1], variant=1, instr_tag_before='', instr_tag_after=''):
        """
        Extracts answers from the model given a question and possible answers.
        """
        if values is None:
            values = [1] * len(answers)
        input_texts = [f'{tokenizer.pad_token}{instr_tag_before}{question}{instr_tag_after} {answer}' for answer in answers]
        input_ids, attention_mask = tokenizer(input_texts, padding=True, return_tensors="pt").values()
        input_ids = input_ids.to(torch.device('cuda'))
        attention_mask = attention_mask.to(torch.device('cuda'))
        attention_mask[:,0] = 0
        question_length = len(tokenizer(question)['input_ids'])
        model = model.to(input_ids.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        result_df = pd.DataFrame()
        for temp in temps:
            if variant == 1:
                probs = torch.log_softmax(outputs.logits / temp, dim=-1).detach()
            else:
                probs = outputs.logits.detach()
            _ids = input_ids[:, 1:]
            gen_probs = torch.gather(probs, 2, _ids[:, :, None]).squeeze(-1)
            if variant == 2:
                gen_probs = torch.log_softmax(gen_probs / temp, dim=0)
            if variant == 1 or variant == 2:
                batch_df = pd.DataFrame()
                for input_sentence, input_probs, answer, value in zip(_ids, gen_probs, answers, values):
                    text_sequence = []
                    for token, p in list(zip(input_sentence, input_probs))[question_length:]:
                        if token not in tokenizer.all_special_ids and token != 29871:
                            text_sequence.append((tokenizer.decode(token), p.item()))
                    answer_df = pd.DataFrame(text_sequence, columns=['token', 'prob'])
                    answer_df['answer'] = answer
                    answer_df['value'] = value
                    batch_df = pd.concat([batch_df, answer_df])
                batch_df = batch_df.drop(columns=['token']).groupby(['answer', 'value']).mean().reset_index()
            batch_df['temp'] = temp
            batch_df['answer'] = batch_df['answer'].replace('</s>', 'N/A')
            result_df = pd.concat([result_df, batch_df])
        one_word_prompt = "Complete the following with a one-word adjective that aptly describes your personality: "
        combined_one_word_question = one_word_prompt + question + " is"
        question_input_ids = tokenizer(
            f'{tokenizer.pad_token}{combined_one_word_question}', return_tensors="pt"
        ).input_ids.to(torch.device('cuda'))
        attention_mask = question_input_ids.ne(tokenizer.pad_token_id).long()
        generated_outputs = model.model.generate(
            question_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            num_return_sequences=1,
            do_sample=False,    # Greedy decoding for deterministic output, if False
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.5,    # Sampling temperature
            # top_p=0.95,
        )
        decoded_answer = tokenizer.decode(generated_outputs[0], skip_special_tokens=True).strip()
        # print(f"Model's continuation of the question: {decoded_answer}")

        # Normalize the probabilities (for variant 1)
        if variant == 1:
            log_probs = result_df['prob'].values
            log_probs_shifted = log_probs - np.max(log_probs)
            probs = np.exp(log_probs_shifted)
            normalized_probs = probs / probs.sum()
            result_df['norm_probs'] = normalized_probs
        return result_df.reset_index(drop=True)