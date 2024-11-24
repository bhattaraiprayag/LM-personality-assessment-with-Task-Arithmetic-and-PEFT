# src/experiment_tracker.py
"""
Module for tracking experiments, updating metadata, and saving
evaluation results for language model assessments.
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
    Class to manage experiment metadata and results, including
    updating Excel files and normalizing results.

    Attributes:
        metadata_file (str): Path to the experiment metadata JSON file.
        output_dir (str): Directory where outputs are saved.
        excel_file (str): Path to the Excel file summarizing experiments.
    """

    def __init__(self, metadata_file, output_dir,
                 excel_file="experiment_results.xlsx") -> None:
        """
        Initializes the ExperimentTracker with metadata and output paths.

        Args:
            metadata_file (str): Path to the metadata JSON file.
            output_dir (str): Directory for output files.
            excel_file (str): Filename for the Excel summary file.
        """
        self.metadata_file = metadata_file
        self.output_dir = output_dir
        self.excel_file = os.path.join(self.output_dir, excel_file)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """
        Loads experiment metadata from a JSON file.

        Returns:
            dict: Dictionary containing experiment metadata.
        """
        with open(self.metadata_file, "r") as file:
            return json.load(file)

    def _extract_args(self, experiment_data: dict) -> tuple:
        """
        Extracts arguments and results from experiment data.

        Args:
            experiment_data (dict): Data for a single experiment.

        Returns:
            Tuple[dict, dict]: Tuple containing arguments and results
                dictionaries.
        """
        args = experiment_data.copy()
        results = args.pop("results", None)
        return args, results

    def update_excel(self) -> None:
        """
        Updates the Excel file with experiment arguments and generated
        commands.
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

    def _generate_command(self, args: dict) -> str:
        """
        Generates a command-line string for running an experiment based
        on its arguments.

        Args:
            args (dict): Dictionary of experiment arguments.

        Returns:
            str: Command-line string.
        """
        command = f"python start_experiment.py --dataset {args['dataset']} --split {args['split']} "
        command += f"--output {args['output']} --model_name {args['model_name']} --seed {args['seed']} "
        command += f"--epochs {args['epochs']} --batch_size {args['batch_size']} --lr {args['lr']} "
        if args.get("use_peft"):
            command += f"--use_peft {args['use_peft']} "
        if args.get("scale_peft"):
            command += f"--scale_peft {args['scale_peft']} "
        return command

    def save_evaluation_results(self) -> None:
        """
        Saves pre- and post-finetuning evaluation results to CSV files.
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

    def _save_pre_post_results(self, exp_id: str, eval_data: dict,
                               phase: str) -> None:
        """
        Saves evaluation results for a specific phase (pre or post finetuning).

        Args:
            exp_id (str): Experiment identifier.
            eval_data (dict): Evaluation data.
            phase (str): Phase of evaluation ('pre' or 'post').
        """
        output_path = os.path.join(self.output_dir, exp_id)
        os.makedirs(output_path, exist_ok=True)
        df = pd.DataFrame(eval_data)
        answer_to_trait_key = {v: k for k, v in OCEAN_TRAIT_ANSWER_KEYS.items()}

        def extract_trait_polarity(answer):
            """
            Extracts trait and polarity from an answer.

            Args:
                answer (str): Answer text.

            Returns:
                pd.Series: Series containing trait, polarity, and trait code.
            """
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

    def perform_horizontal_normalization(self) -> None:
        """
        Performs horizontal normalization of probabilities across experiments
        for the same data split.
        """
        for phase in ["pre", "post"]:
            splits = set(exp_data["split"] for exp_data in self.metadata.values())
            for split in splits:
                # Collect base results across experiments for the same split
                base_results_list = []
                for exp_id, exp_data in self.metadata.items():
                    if exp_data["split"] == split:
                        output_path = os.path.join(self.output_dir, exp_id)
                        base_results_path = os.path.join(
                            output_path, f"personality_eval_{phase}_base_results.csv"
                        )
                        if os.path.exists(base_results_path):
                            base_results_df = pd.read_csv(base_results_path)
                            base_results_df["exp_id"] = exp_id
                            base_results_list.append(base_results_df)
                if base_results_list:
                    all_base_results_df = pd.concat(
                        base_results_list, ignore_index=True
                    )
                    # Compute min and max probabilities for each temp, trait, polarity
                    group_cols = ["temp", "trait", "polarity"]
                    min_max_df = (
                        all_base_results_df.groupby(group_cols)["prob"]
                        .agg(["min", "max"])
                        .reset_index()
                    )
                    # Merge min and max back to base results
                    all_base_results_df = all_base_results_df.merge(
                        min_max_df, on=group_cols, how="left"
                    )

                    # Compute horizontally normalized probabilities
                    def compute_hor_norm_prob(row):
                        if row["max"] != row["min"]:
                            return (row["prob"] - row["min"]) / (
                                row["max"] - row["min"]
                            )
                        else:
                            return 0.0

                    all_base_results_df["hor_norm_prob"] = all_base_results_df.apply(
                        compute_hor_norm_prob, axis=1
                    )
                    # Compute net_then_hor_norm across experiments
                    net_probs_list = []
                    for exp_id in all_base_results_df["exp_id"].unique():
                        exp_df = all_base_results_df[
                            all_base_results_df["exp_id"] == exp_id
                        ]
                        for temp in exp_df["temp"].unique():
                            temp_df = exp_df[exp_df["temp"] == temp]
                            for trait in temp_df["trait"].unique():
                                trait_df = temp_df[temp_df["trait"] == trait]
                                pos_trait = trait_df[trait_df["polarity"] == "+"]
                                neg_trait = trait_df[trait_df["polarity"] == "-"]
                                if not pos_trait.empty and not neg_trait.empty:
                                    net_prob = (
                                        pos_trait["prob"].iloc[0]
                                        - neg_trait["prob"].iloc[0]
                                    )
                                else:
                                    net_prob = None
                                net_probs_list.append(
                                    {
                                        "exp_id": exp_id,
                                        "temp": temp,
                                        "trait": trait,
                                        "net_prob": net_prob,
                                    }
                                )
                    net_probs_df = pd.DataFrame(net_probs_list)
                    # Compute min and max net_prob for each temp and trait across experiments
                    net_min_max_df = (
                        net_probs_df.groupby(["temp", "trait"])["net_prob"]
                        .agg(["min", "max"])
                        .reset_index()
                    )
                    # Merge min and max back to net_probs_df
                    net_probs_df = net_probs_df.merge(
                        net_min_max_df, on=["temp", "trait"], how="left"
                    )

                    # Compute net_then_hor_norm
                    def compute_net_then_hor_norm(row):
                        if row["max"] != row["min"]:
                            return (row["net_prob"] - row["min"]) / (
                                row["max"] - row["min"]
                            )
                        else:
                            return 0.0

                    net_probs_df["net_then_hor_norm"] = net_probs_df.apply(
                        compute_net_then_hor_norm, axis=1
                    )
                    # Update net_results files for each experiment
                    for exp_id in net_probs_df["exp_id"].unique():
                        output_path = os.path.join(self.output_dir, exp_id)
                        net_results_path = os.path.join(
                            output_path, f"personality_eval_{phase}_net_results.csv"
                        )
                        if os.path.exists(net_results_path):
                            net_results_df = pd.read_csv(net_results_path)
                            exp_net_probs_df = net_probs_df[
                                net_probs_df["exp_id"] == exp_id
                            ]
                            # Compute hor_norm_then_net for this experiment
                            exp_df = all_base_results_df[
                                all_base_results_df["exp_id"] == exp_id
                            ]
                            hor_norm_then_net_list = []
                            for temp in exp_df["temp"].unique():
                                temp_df = exp_df[exp_df["temp"] == temp]
                                for trait in temp_df["trait"].unique():
                                    trait_df = temp_df[temp_df["trait"] == trait]
                                    pos_trait = trait_df[trait_df["polarity"] == "+"]
                                    neg_trait = trait_df[trait_df["polarity"] == "-"]
                                    if not pos_trait.empty and not neg_trait.empty:
                                        net_prob = (
                                            pos_trait["hor_norm_prob"].iloc[0]
                                            - neg_trait["hor_norm_prob"].iloc[0]
                                        )
                                    else:
                                        net_prob = None
                                    hor_norm_then_net_list.append(
                                        {
                                            "temp": temp,
                                            "trait": trait,
                                            "hor_norm_then_net": net_prob,
                                        }
                                    )
                            hor_norm_then_net_df = pd.DataFrame(hor_norm_then_net_list)
                            # Merge new columns into net_results_df
                            net_results_df = net_results_df.merge(
                                hor_norm_then_net_df, on=["temp", "trait"], how="left"
                            )
                            net_results_df = net_results_df.merge(
                                exp_net_probs_df[
                                    ["temp", "trait", "net_then_hor_norm"]
                                ],
                                on=["temp", "trait"],
                                how="left",
                            )
                            # Reorder columns
                            net_results_df = net_results_df[
                                [
                                    "temp",
                                    "trait",
                                    "norm_then_net",
                                    "net_then_norm",
                                    "hor_norm_then_net",
                                    "net_then_hor_norm",
                                ]
                            ]
                            # Save updated net_results_df
                            net_results_df.to_csv(net_results_path, index=False)
                            print(
                                f"Updated {phase}-finetuning net results for experiment {exp_id} at {net_results_path}."
                            )

    def run(self) -> None:
        """
        Runs the full tracking process: updating Excel summaries,
        saving evaluation results, and normalizing data.
        """
        self.update_excel()
        self.save_evaluation_results()
        self.perform_horizontal_normalization()


if __name__ == "__main__":
    OUTPUT_DIR = "../outputs"
    METADATA_FILE = f"{OUTPUT_DIR}/experiment_metadata_updated.json"
    tracker = ExperimentTracker(METADATA_FILE, OUTPUT_DIR)
    tracker.run()
