import os
import json
import math
import pandas as pd
from collections import defaultdict
from experiment_config import OCEAN_TRAITS, OCEAN_TRAIT_ANSWER_KEYS


answer_to_trait_polarity = {}
for key, val in OCEAN_TRAIT_ANSWER_KEYS.items():
    trait_letter, polarity = key.split()
    answer_to_trait_polarity[val] = (trait_letter, polarity.strip("()"))

def process_personality_json(
    json_path: str,
    results_output_dir: str = "results"
):
    with open(json_path, "r", encoding="utf-8") as f:
        all_experiments = json.load(f)

    experiments_by_split = defaultdict(list)
    for exp_id, exp_data in all_experiments.items():
        split_name = exp_data["split"]
        experiments_by_split[split_name].append((exp_id, exp_data))

    for personality_vector, exp_list in experiments_by_split.items():
        print(f"Processing personality vector: {personality_vector}")

        data_rows = []
        min_nonzero_prob = float("inf")

        for (exp_id, exp_data) in exp_list:
            use_peft = exp_data["use_peft"]
            exp_type = "base" if use_peft is None else use_peft
            
            seed = exp_data.get("seed", 0)

            results = exp_data["results"]

            # --- Handle PRE-FT ---
            if "personality_eval_pre" in results:
                pre_eval_dict = results["personality_eval_pre"]
                for scale_key, answer_list in pre_eval_dict.items():
                    # Only keep "pre" if it's baseline:
                    if use_peft is None:
                        scale_val = parse_scale_key(scale_key)
                        for ans_item in answer_list:
                            prob = ans_item["prob"]
                            if prob > 0 and prob < min_nonzero_prob:
                                min_nonzero_prob = prob

                            answer_text = ans_item["answer"]
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
                            })

            # --- Handle POST-FT ---
            if "personality_eval_post" in results:
                post_eval_dict = results["personality_eval_post"]
                for scale_key, answer_list in post_eval_dict.items():
                    scale_val = parse_scale_key(scale_key)
                    for ans_item in answer_list:
                        prob = ans_item["prob"]
                        if prob > 0 and prob < min_nonzero_prob:
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
                        })

        if min_nonzero_prob == float("inf"):
            min_nonzero_prob = 1e-64

        safe_min = min_nonzero_prob / 10.0

        df = pd.DataFrame(data_rows)

        df["prob"] = df["prob"].apply(lambda p: p if p > 0 else safe_min)
        df["log_prob"] = df["prob"].apply(math.log)

        df.loc[df["exp_type"] == "base", "scale"] = None
        df["scale"] = df["scale"].apply(lambda x: 'None' if pd.isna(x) else x)

        df["trait"] = df["trait"].str.strip().str.replace(r"[,.\s]+$", "", regex=True)

        grouped_logprob = df.groupby(["seed", "answer"])["log_prob"]
        min_map = grouped_logprob.transform("min")
        max_map = grouped_logprob.transform("max")
        eps = 1e-1000

        df["norm_prob"] = (df["log_prob"] - min_map) / ((max_map - min_map) + eps)

        processed_results = df[
            [
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
            ["seed", "phase", "exp_type", "temp", "scale", "trait", "polarity", "norm_prob"]
        ].copy()

        net_rows = []
        group_cols = ["seed", "phase", "exp_type", "temp", "scale", "trait"]
        for grp_vals, grp_df in df_for_net.groupby(group_cols):
            plus_df = grp_df[grp_df["polarity"] == "+"]
            minus_df = grp_df[grp_df["polarity"] == "-"]
            if plus_df.empty or minus_df.empty:
                continue

            norm_net = plus_df["norm_prob"].mean() - minus_df["norm_prob"].mean()

            net_rows.append({
                "seed": grp_vals[0],
                "phase": grp_vals[1],
                "exp_type": grp_vals[2],
                "temp": grp_vals[3],
                "scale": grp_vals[4],
                "trait": grp_vals[5],
                "norm_net": norm_net,
            })

        net_results = pd.DataFrame(net_rows)

        subfolder = os.path.join(results_output_dir, personality_vector)
        os.makedirs(subfolder, exist_ok=True)

        processed_csv_path = os.path.join(subfolder, "processed_results.csv")
        net_csv_path = os.path.join(subfolder, "net_results.csv")

        processed_results.to_csv(processed_csv_path, index=False)
        net_results.to_csv(net_csv_path, index=False)

        print(f"  --> Saved processed_results to: {processed_csv_path}")
        print(f"  --> Saved net_results to:      {net_csv_path}")

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

if __name__ == "__main__":
    OUTPUT_DIR = "outputs"
    RESULTS_DIR = f"{OUTPUT_DIR}/results"
    JSON_PATH = f"{OUTPUT_DIR}/experiment_metadata_latest.json"
    # JSON_PATH = f"{OUTPUT_DIR}/experiment_metadata_wo_fullstop.json"
    process_personality_json(
        json_path=JSON_PATH,
        results_output_dir=RESULTS_DIR
    )
