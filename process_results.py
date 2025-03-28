import json
import math
import os
from collections import defaultdict

import pandas as pd

from experiment_config import (OCEAN_TRAIT_ANSWER_KEYS, OCEAN_TRAITS,
                               PANAS_X_TRAIT_SUBCLASSES)

# Personality mapping: answer -> (trait_letter, polarity)
answer_to_trait_polarity = {}
for key, val in OCEAN_TRAIT_ANSWER_KEYS.items():
    trait_letter, polarity = key.split()
    answer_to_trait_polarity[val] = (trait_letter, polarity.strip("()"))

# Emotion mapping: answer -> emotion_class
answer_to_emotion_class = {}
for emosubclass, answers_list in PANAS_X_TRAIT_SUBCLASSES.items():
    for ans in answers_list:
        # Lowercase comparison is safer, but your code normalizes them anyway.
        # Ensure the text in the JSON is consistent with these keys (e.g. 'angry.', 'hostile.', etc.)
        # If needed, strip trailing periods or adapt to your actual stored 'answer' strings.
        normalized = ans.lower().replace(".", "").strip()
        answer_to_emotion_class[normalized] = emosubclass


def parse_scale_key(scale_key: str):
    prefix = "scale_"
    if scale_key.startswith(prefix):
        val_str = scale_key[len(prefix):]
        if val_str.lower() == "none":
            return None
        else:
            try:
                return float(val_str)
            except ValueError:
                return None
    return None


def process_results(json_path: str, results_output_dir: str = "results"):
    """
    Reads experiment_metadata_norm.json containing both 'pandora' (personality) and
    'emotion' (PANAS-X-based) evaluations, processes them appropriately, and saves
    up to four CSVs:
      - all_processed_results_personality.csv
      - all_net_results_personality.csv
      - all_processed_results_emotion.csv
      - all_net_results_emotion.csv
    Only writes the personality files if pandora experiments exist, only writes
    the emotion files if emotion experiments exist.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        all_experiments = json.load(f)

    # We'll group experiments by (dataset, split), so we can process them in the same spirit
    # that your old code used for personality "split" data. 
    # For minimal changes, we keep the "split" grouping, but we separate by dataset as well.
    # So effectively we have two top-level lumps: pandora vs emotion.
    experiments_by_dataset_and_split = defaultdict(lambda: defaultdict(list))

    for exp_id, exp_data in all_experiments.items():
        dataset_type = exp_data.get("dataset", None)  # 'pandora' or 'emotion'
        if not dataset_type:
            # If for some reason 'dataset' is missing, skip or treat as unknown
            continue
        split_name = exp_data.get("split", "unknown_split")
        # Now store this experiment under that dataset + that split.
        experiments_by_dataset_and_split[dataset_type][split_name].append((exp_id, exp_data))

    # We will gather results in separate data frames for personality vs emotion
    # to produce separate CSV outputs.
    personality_processed_rows = []
    personality_net_rows = []
    emotion_processed_rows = []
    emotion_net_rows = []

    # ================================================================
    # Helper function to compute a "safe_min" replacement for zero prob
    # so we avoid math.log(0).
    # ================================================================
    def get_safe_min_prob(min_nonzero):
        if min_nonzero == float("inf"):
            return 1e-64
        return min_nonzero / 10.0

    # ----------------------------------------------------------------
    # PROCESS PERSONALITY (PANDORA) EXACTLY LIKE ORIGINAL, 
    # but referencing custom_eval_pre/post
    # ----------------------------------------------------------------
    if "pandora" in experiments_by_dataset_and_split:
        personality_exps_by_split = experiments_by_dataset_and_split["pandora"]

        for personality_vector, exp_list in personality_exps_by_split.items():
            print(f"[PERSONALITY] Processing personality vector: {personality_vector}")

            # Same logic for "short_vector" if you want
            short_vector = personality_vector

            data_rows = []
            min_nonzero_prob = float("inf")

            # Gather all rows from all experiments that share this 'split' (vector).
            for (exp_id, exp_data) in exp_list:
                use_peft = exp_data.get("use_peft", None)
                exp_type = "base" if use_peft is None else use_peft
                seed = exp_data.get("seed", 0)

                results = exp_data.get("results", {})
                # Instead of "personality_eval_pre", we now check "custom_eval_pre"
                if "custom_eval_pre" in results or "personality_eval_pre" in results:
                # if "custom_eval_pre" in results:
                    # pre_eval_dict = results["custom_eval_pre"]
                    pre_eval_dict = results["custom_eval_pre"] if "custom_eval_pre" in results else results["personality_eval_pre"]
                    for scale_key, answer_list in pre_eval_dict.items():
                        scale_val = parse_scale_key(scale_key)
                        for ans_item in answer_list:
                            prob = ans_item["prob"]
                            if 0 < prob < min_nonzero_prob:
                                min_nonzero_prob = prob

                            # Old code used ans_item["answer"] to find trait & polarity.
                            answer_text = ans_item["answer"]
                            # We need to handle the dictionary. If not found, fallback:
                            trait_letter, polarity = answer_to_trait_polarity.get(
                                answer_text, ("?", "?")
                            )
                            trait_name = OCEAN_TRAITS.get(trait_letter, "Unknown")

                            data_rows.append({
                                "seed": seed,
                                "phase": "pre",
                                "exp_type": exp_type,
                                "scale": scale_val,
                                "temp": ans_item["temp"],
                                "answer": answer_text,
                                "trait": trait_name,
                                "polarity": polarity,
                                "prob": prob,
                                "vector": short_vector,
                            })

                # if "custom_eval_post" in results:
                if "custom_eval_post" in results or "personality_eval_post" in results:
                    # post_eval_dict = results["custom_eval_post"]
                    post_eval_dict = results["custom_eval_post"] if "custom_eval_post" in results else results["personality_eval_post"]
                    for scale_key, answer_list in post_eval_dict.items():
                        scale_val = parse_scale_key(scale_key)
                        for ans_item in answer_list:
                            prob = ans_item["prob"]
                            if 0 < prob < min_nonzero_prob:
                                min_nonzero_prob = prob

                            answer_text = ans_item["answer"]
                            trait_letter, polarity = answer_to_trait_polarity.get(
                                answer_text, ("?", "?")
                            )
                            trait_name = OCEAN_TRAITS.get(trait_letter, "Unknown")

                            data_rows.append({
                                "seed": seed,
                                "phase": "post",
                                "exp_type": exp_type,
                                "scale": scale_val,
                                "temp": ans_item["temp"],
                                "answer": answer_text,
                                "trait": trait_name,
                                "polarity": polarity,
                                "prob": prob,
                                "vector": short_vector,
                            })

            if not data_rows:
                # No data at all for this personality vector, skip.
                continue

            safe_min = get_safe_min_prob(min_nonzero_prob)

            df = pd.DataFrame(data_rows)
            df["prob"] = df["prob"].apply(lambda p: p if p > 0 else safe_min)
            df["log_prob"] = df["prob"].apply(math.log)

            # Keep old logic for exp_type, scale, etc.
            df.loc[df["exp_type"] == "base", "scale"] = None
            df["scale"] = df["scale"].apply(lambda x: 'None' if pd.isna(x) else x)

            df["trait"] = df["trait"].str.strip().str.replace(r"[,.\s]+$", "", regex=True)

            # The original min/max log_prob approach per (seed, answer):
            grouped_logprob = df.groupby(["seed", "answer"])["log_prob"]
            min_map = grouped_logprob.transform("min")
            max_map = grouped_logprob.transform("max")
            eps = 1e-1000
            df["norm_prob"] = (df["log_prob"] - min_map) / ((max_map - min_map) + eps)

            processed_results = df[
                [
                    "vector",
                    "seed",
                    "phase",
                    "exp_type",
                    "temp",
                    "scale",
                    "answer",
                    "trait",
                    "polarity",
                    "prob",
                    "log_prob",
                    "norm_prob",
                ]
            ].copy()

            df_for_net = processed_results[
                [
                    "vector",
                    "seed",
                    "phase",
                    "exp_type",
                    "temp",
                    "scale",
                    "trait",
                    "polarity",
                    "norm_prob"
                ]
            ].copy()

            # Net approach: plus minus
            net_rows = []
            group_cols = [
                "vector",
                "seed",
                "phase",
                "exp_type",
                "temp",
                "scale",
                "trait",
            ]
            for grp_vals, grp_df in df_for_net.groupby(group_cols):
                plus_df = grp_df[grp_df["polarity"] == "+"]
                minus_df = grp_df[grp_df["polarity"] == "-"]
                # If we don't have both polarities, skip
                if plus_df.empty or minus_df.empty:
                    continue
                norm_net = plus_df["norm_prob"].mean() - minus_df["norm_prob"].mean()
                net_rows.append({
                    "vector": grp_vals[0],
                    "seed": grp_vals[1],
                    "phase": grp_vals[2],
                    "exp_type": grp_vals[3],
                    "temp": grp_vals[4],
                    "scale": grp_vals[5],
                    "trait": grp_vals[6],
                    "norm_net": norm_net,
                })

            net_results = pd.DataFrame(net_rows)

            # Accumulate
            personality_processed_rows.extend(processed_results.to_dict("records"))
            personality_net_rows.extend(net_results.to_dict("records"))

    # ----------------------------------------------------------------
    # PROCESS EMOTION
    # We'll group and average norm_prob by the four classes (ANGER, SADNESS, JOY, OPTIMISM).
    # ----------------------------------------------------------------
    if "emotion" in experiments_by_dataset_and_split:
        emotion_exps_by_split = experiments_by_dataset_and_split["emotion"]

        for emotion_split_name, exp_list in emotion_exps_by_split.items():
            print(f"[EMOTION] Processing split: {emotion_split_name}")

            data_rows = []
            min_nonzero_prob = float("inf")

            for (exp_id, exp_data) in exp_list:
                use_peft = exp_data.get("use_peft", None)
                exp_type = "base" if use_peft is None else use_peft
                seed = exp_data.get("seed", 0)

                results = exp_data.get("results", {})
                # For emotion, everything is in custom_eval_pre or custom_eval_post
                if "custom_eval_pre" in results:
                    pre_eval_dict = results["custom_eval_pre"]
                    for scale_key, answer_list in pre_eval_dict.items():
                        scale_val = parse_scale_key(scale_key)
                        for ans_item in answer_list:
                            prob = ans_item["prob"]
                            if 0 < prob < min_nonzero_prob:
                                min_nonzero_prob = prob

                            answer_text = ans_item["answer"]
                            # Attempt to match to emotion class:
                            # Lowercase and maybe remove trailing period so it lines up
                            # with the keys we stored in answer_to_emotion_class.
                            clean_answer = answer_text.lower().replace(".", "").strip()
                            emosubclass = answer_to_emotion_class.get(clean_answer, "Unknown")

                            data_rows.append({
                                "seed": seed,
                                "phase": "pre",
                                "exp_type": exp_type,
                                "scale": scale_val,
                                "temp": ans_item["temp"],
                                "answer": answer_text,
                                "emotion_class": emosubclass,
                                "prob": prob,
                                "vector": emotion_split_name,
                            })

                if "custom_eval_post" in results:
                    post_eval_dict = results["custom_eval_post"]
                    for scale_key, answer_list in post_eval_dict.items():
                        scale_val = parse_scale_key(scale_key)
                        for ans_item in answer_list:
                            prob = ans_item["prob"]
                            if 0 < prob < min_nonzero_prob:
                                min_nonzero_prob = prob

                            answer_text = ans_item["answer"]
                            clean_answer = answer_text.lower().replace(".", "").strip()
                            emosubclass = answer_to_emotion_class.get(clean_answer, "Unknown")

                            data_rows.append({
                                "seed": seed,
                                "phase": "post",
                                "exp_type": exp_type,
                                "scale": scale_val,
                                "temp": ans_item["temp"],
                                "answer": answer_text,
                                "emotion_class": emosubclass,
                                "prob": prob,
                                "vector": emotion_split_name,
                            })

            if not data_rows:
                continue

            safe_min = get_safe_min_prob(min_nonzero_prob)
            df = pd.DataFrame(data_rows)
            df["prob"] = df["prob"].apply(lambda p: p if p > 0 else safe_min)
            df["log_prob"] = df["prob"].apply(math.log)

            df.loc[df["exp_type"] == "base", "scale"] = None
            df["scale"] = df["scale"].apply(lambda x: 'None' if pd.isna(x) else x)

            # We do the same min/max transform for norm_prob
            grouped_logprob = df.groupby(["seed", "answer"])["log_prob"]
            min_map = grouped_logprob.transform("min")
            max_map = grouped_logprob.transform("max")
            eps = 1e-1000
            df["norm_prob"] = (df["log_prob"] - min_map) / ((max_map - min_map) + eps)

            processed_results = df[
                [
                    "vector",
                    "seed",
                    "phase",
                    "exp_type",
                    "temp",
                    "scale",
                    "answer",
                    "emotion_class",
                    "prob",
                    "log_prob",
                    "norm_prob",
                ]
            ].copy()

            # For "net" in emotion, we simply AVERAGE across all items belonging to that emotion_class.
            # We do not have polarity (+/-).
            df_for_net = processed_results[
                [
                    "vector",
                    "seed",
                    "phase",
                    "exp_type",
                    "temp",
                    "scale",
                    "emotion_class",
                    "norm_prob"
                ]
            ].copy()

            net_rows = []
            group_cols = [
                "vector",
                "seed",
                "phase",
                "exp_type",
                "temp",
                "scale",
                "emotion_class",
            ]
            for grp_vals, grp_df in df_for_net.groupby(group_cols):
                avg_norm_prob = grp_df["norm_prob"].mean()
                net_rows.append({
                    "vector": grp_vals[0],
                    "seed": grp_vals[1],
                    "phase": grp_vals[2],
                    "exp_type": grp_vals[3],
                    "temp": grp_vals[4],
                    "scale": grp_vals[5],
                    "emotion_class": grp_vals[6],
                    # We'll store the average as 'norm_net'
                    "norm_net": avg_norm_prob,
                })

            net_results = pd.DataFrame(net_rows)

            # Accumulate
            emotion_processed_rows.extend(processed_results.to_dict("records"))
            emotion_net_rows.extend(net_results.to_dict("records"))

    # ----------------------------------------------------------------
    # Finally, write out the CSVs. We only produce them if we have data.
    # ----------------------------------------------------------------
    os.makedirs(results_output_dir, exist_ok=True)

    # Personality CSVs
    if personality_processed_rows:
        all_processed_df = pd.DataFrame(personality_processed_rows)
        all_net_df = pd.DataFrame(personality_net_rows)

        processed_csv_path = os.path.join(results_output_dir, "all_processed_results_personality.csv")
        net_csv_path = os.path.join(results_output_dir, "all_net_results_personality.csv")
        all_processed_df.to_csv(processed_csv_path, index=False)
        all_net_df.to_csv(net_csv_path, index=False)
        print(f"[PERSONALITY] --> Saved all_processed_results to: {processed_csv_path}")
        print(f"[PERSONALITY] --> Saved all_net_results to:      {net_csv_path}")
    else:
        print("No personality (pandora) experiments found.")

    # Emotion CSVs
    if emotion_processed_rows:
        all_processed_df = pd.DataFrame(emotion_processed_rows)
        all_net_df = pd.DataFrame(emotion_net_rows)

        processed_csv_path = os.path.join(results_output_dir, "all_processed_results_emotion.csv")
        net_csv_path = os.path.join(results_output_dir, "all_net_results_emotion.csv")
        all_processed_df.to_csv(processed_csv_path, index=False)
        all_net_df.to_csv(net_csv_path, index=False)
        print(f"[EMOTION] --> Saved all_processed_results to: {processed_csv_path}")
        print(f"[EMOTION] --> Saved all_net_results to:      {net_csv_path}")
    else:
        print("No emotion experiments found.")


if __name__ == "__main__":
    OUTPUT_DIR = "outputs"
    RESULTS_DIR = f"{OUTPUT_DIR}/results"
    JSON_PATH = f"{OUTPUT_DIR}/experiment_metadata_norm.json"

    process_results(
        json_path=JSON_PATH,
        results_output_dir=RESULTS_DIR
    )