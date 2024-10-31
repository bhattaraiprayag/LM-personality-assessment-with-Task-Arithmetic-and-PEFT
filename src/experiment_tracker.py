# src/experiment_tracker.py

"""
Tracks experiment metadata and results, and updates an Excel sheet with the data.
"""

import json
import os

import pandas as pd

OCEAN_TRAIT_ANSWER_KEYS = {
    "O (+)": "has an active imagination.",
    "O (-)": "has few artistic interests.",
    "C (+)": "does a thorough job.",
    "C (-)": "tends to be lazy.",
    "E (+)": "is outgoing, sociable.",
    "E (-)": "is reserved.",
    "A (+)": "is generally trusting.",
    "A (-)": "tends to find fault with others.",
    "N (+)": "is relaxed, handles stress well.",
    "N (-)": "gets nervous easily.",
}

OCEAN_TRAITS = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}


class ExperimentTracker:
    """
    Tracks experiment metadata and results, and updates an Excel sheet with the data.
    """

    def __init__(self, metadata_file, output_dir, excel_file="experiment_results.xlsx"):
        self.metadata_file = metadata_file
        self.output_dir = output_dir
        self.excel_file = os.path.join(self.output_dir, excel_file)
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """
        Load the experiment metadata from the JSON file.
        """
        with open(self.metadata_file, "r") as file:
            return json.load(file)

    def _extract_args(self, experiment_data):
        """
        Extract arguments from the experiment data for the Excel sheet.
        """
        args = experiment_data.copy()
        results = args.pop("results", None)
        return args, results

    def update_excel(self):
        """
        Update the Excel file with experiment arguments.
        """
        records = []
        for exp_id, exp_data in self.metadata.items():
            args, _ = self._extract_args(exp_data)
            command = self._generate_command(args)
            args["id"] = exp_id
            args["command"] = command
            records.append(args)
        df = pd.DataFrame(records)
        if os.path.exists(self.excel_file):
            existing_df = pd.read_excel(self.excel_file)
            df = pd.concat([existing_df, df]).drop_duplicates(subset="id", keep="last")
        df.to_excel(self.excel_file, index=False)
        print(f"Updated {self.excel_file} with experiment data.")

    def _generate_command(self, args):
        """
        Generate the Python command that corresponds to the experiment.
        """
        command = f"python start_experiment.py --dataset {args['dataset']} --split {args['split']} "
        command += f"--output {args['output']} --model_name {args['model_name']} --seed {args['seed']} "
        command += f"--epochs {args['epochs']} --batch_size {args['batch_size']} --lr {args['lr']} "
        if args.get("use_peft"):
            command += f"--use_peft {args['use_peft']} "
        if args.get("scale_peft"):
            command += f"--scale_peft {args['scale_peft']} "
        return command

    def save_evaluation_results(self):
        """
        Save pre- and post-finetuning personality evaluation results as separate files.
        """
        for exp_id, exp_data in self.metadata.items():
            _, results = self._extract_args(exp_data)
            if results:
                self._save_pre_post_results(
                    exp_id, results["personality_eval_pre"], "pre"
                )
                self._save_pre_post_results(
                    exp_id, results["personality_eval_post"], "post"
                )

    def _save_pre_post_results(self, exp_id, eval_data, phase):
        """
        Save the evaluation results for pre- or post-finetuning.
        """
        output_path = os.path.join(self.output_dir, exp_id)
        os.makedirs(output_path, exist_ok=True)
        df = pd.DataFrame(eval_data)
        answer_to_trait_key = {v: k for k, v in OCEAN_TRAIT_ANSWER_KEYS.items()}

        def extract_trait_polarity(answer):
            trait_key = answer_to_trait_key.get(answer)
            if trait_key:
                trait_code, polarity = trait_key.split()
                trait_code = trait_code.strip()
                polarity = polarity.strip("()")
                trait = OCEAN_TRAITS[trait_code]
                return pd.Series(
                    {"trait": trait, "polarity": polarity, "trait_code": trait_code}
                )
            return pd.Series({"trait": None, "polarity": None, "trait_code": None})

        trait_polarity_df = df["answer"].apply(extract_trait_polarity)
        df = pd.concat([df, trait_polarity_df], axis=1)

        base_results_columns = [
            "answer",
            "value",
            "prob",
            "temp",
            "norm_probs",
            "trait",
            "polarity",
        ]
        base_results_df = df[base_results_columns]
        base_results_path = os.path.join(
            output_path, f"personality_eval_{phase}_base_results.csv"
        )
        base_results_df.to_csv(base_results_path, index=False)
        net_results_list = []
        for temp in df["temp"].unique():
            temp_df = df[df["temp"] == temp].copy()
            # norm_then_net
            prob_min = temp_df["prob"].min()
            prob_max = temp_df["prob"].max()
            if prob_max != prob_min:
                temp_df["norm_prob"] = (temp_df["prob"] - prob_min) / (
                    prob_max - prob_min
                )
            else:
                temp_df["norm_prob"] = 0.0
            norm_then_net = {}
            for trait_code in OCEAN_TRAITS.keys():
                pos_trait = temp_df[
                    (temp_df["trait_code"] == trait_code) & (temp_df["polarity"] == "+")
                ]
                neg_trait = temp_df[
                    (temp_df["trait_code"] == trait_code) & (temp_df["polarity"] == "-")
                ]
                if not pos_trait.empty and not neg_trait.empty:
                    net_prob = (
                        pos_trait["norm_prob"].iloc[0] - neg_trait["norm_prob"].iloc[0]
                    )
                else:
                    net_prob = None
                norm_then_net[trait_code] = net_prob
            # net_then_norm
            net_probs = {}
            for trait_code in OCEAN_TRAITS.keys():
                pos_trait = temp_df[
                    (temp_df["trait_code"] == trait_code) & (temp_df["polarity"] == "+")
                ]
                neg_trait = temp_df[
                    (temp_df["trait_code"] == trait_code) & (temp_df["polarity"] == "-")
                ]
                if not pos_trait.empty and not neg_trait.empty:
                    net_prob = pos_trait["prob"].iloc[0] - neg_trait["prob"].iloc[0]
                else:
                    net_prob = None
                net_probs[trait_code] = net_prob
            net_probs_values = [v for v in net_probs.values() if v is not None]
            net_probs_min = min(net_probs_values) if net_probs_values else 0.0
            net_probs_max = max(net_probs_values) if net_probs_values else 1.0
            net_then_norm = {}
            for trait_code, net_prob in net_probs.items():
                if net_prob is not None and net_probs_max != net_probs_min:
                    normalized_net_prob = (net_prob - net_probs_min) / (
                        net_probs_max - net_probs_min
                    )
                else:
                    normalized_net_prob = 0.0
                net_then_norm[trait_code] = normalized_net_prob

            for trait_code in OCEAN_TRAITS.keys():
                net_result = {
                    "temp": temp,
                    "trait": OCEAN_TRAITS[trait_code],
                    "norm_then_net": norm_then_net.get(trait_code),
                    "net_then_norm": net_then_norm.get(trait_code),
                }
                net_results_list.append(net_result)
        net_results_df = pd.DataFrame(net_results_list)
        net_results_path = os.path.join(
            output_path, f"personality_eval_{phase}_net_results.csv"
        )
        net_results_df.to_csv(net_results_path, index=False)
        print(f"Saved {phase}-finetuning base results to {base_results_path}.")
        print(f"Saved {phase}-finetuning net results to {net_results_path}.")

    def run(self):
        """
        Run the experiment tracker to update Excel and save results.
        """
        self.update_excel()
        self.save_evaluation_results()


if __name__ == "__main__":
    OUTPUT_DIR = "../outputs/"
    METADATA_FILE = f"{OUTPUT_DIR}valid_experiment_metadata.json"
    tracker = ExperimentTracker(METADATA_FILE, OUTPUT_DIR)
    tracker.run()
