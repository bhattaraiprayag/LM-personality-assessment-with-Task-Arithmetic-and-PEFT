# src/eval_results_manager.py
"""Store evaluation results in per-run and consolidated CSV files."""

import json
import os
from typing import Any

import pandas as pd

MASTER_FILES = {
    "mid": "combined_mid_epoch_results.csv",
    "post": "combined_post_epoch_results.csv",
}


class EvalResultsManager:
    """Tiny writer that keeps two global CSVs *and* a per‑run copy."""

    COLS = [
        "exp_id",
        "phase",
        "use_peft",
        "lora_scale",
        "epoch",
        "step",
        "inventory",
        "item",
        "likert_in_prompt",
        1,
        2,
        3,
        4,
        5,
    ]

    @staticmethod
    def append_rows(
        *,
        df: pd.DataFrame,
        phase: str,
        inventory: str,
        exp_id: str,
        output_dir: str,
        use_peft: str,
        lora_scale: str | float,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """Append a batch of scored rows to the master and local result files."""
        df = df.copy()
        df.insert(0, "inventory", inventory)
        df.insert(0, "lora_scale", lora_scale)
        df.insert(0, "use_peft", use_peft)
        df.insert(0, "phase", phase)
        df.insert(0, "exp_id", exp_id)
        df["epoch"] = epoch
        df["step"] = step
        df = df[EvalResultsManager.COLS]

        master_path = os.path.join(output_dir, MASTER_FILES[phase])
        header = not os.path.exists(master_path)
        df.to_csv(master_path, mode="a", header=header, index=False)

        local_dir = os.path.join(output_dir, exp_id, "evals")
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f"{phase}_epoch_results.csv")
        header = not os.path.exists(local_path)
        df.to_csv(local_path, mode="a", header=header, index=False)

    @staticmethod
    def save_custom_eval_results(
        output_dir: str,
        experiment_id: str,
        phase: str,
        eval_type: str,
        question: str,
        answers: list[str],
        results: dict[str, list[dict[str, Any]]],
        epoch: int | None = None,
        step: int | None = None,
    ) -> str:
        """Save custom evaluation results to a CSV file."""
        evals_dir = os.path.join(output_dir, experiment_id, "evals")
        os.makedirs(evals_dir, exist_ok=True)

        if phase == "mid":
            filename = f"custom_eval_{eval_type}_{phase}.csv"
        else:
            filename = f"custom_eval_{eval_type}_{phase}.csv"

        filepath = os.path.join(evals_dir, filename)

        rows = []
        for scale, scale_results in results.items():
            scale_value = scale.replace("scale_", "")
            for result in scale_results:
                row = {
                    "experiment_id": experiment_id,
                    "phase": phase,
                    "eval_type": eval_type,
                    "question": question,
                    "answer": result["answer"],
                    "temperature": result["temp"],
                    "probability": result["prob"],
                    "scale": scale_value,
                }

                if epoch is not None:
                    row["epoch"] = epoch
                if step is not None:
                    row["step"] = step

                rows.append(row)

        new_df = pd.DataFrame(rows)

        if phase == "mid" and os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(filepath, index=False)
        else:
            new_df.to_csv(filepath, index=False)

        return filepath

    @staticmethod
    def compile_mid_epoch_results(
        output_dir: str, experiment_id: str, eval_type: str
    ) -> str | None:
        """Compile legacy mid-epoch JSON outputs into a CSV file."""
        evals_dir = os.path.join(output_dir, experiment_id, "evals")
        mid_file = os.path.join(evals_dir, f"custom_eval_{eval_type}_mid.csv")

        if os.path.exists(mid_file):
            return mid_file

        json_files = []
        for filename in os.listdir(evals_dir):
            if filename.startswith("epoch") and "step" in filename and filename.endswith(".json"):
                json_files.append(os.path.join(evals_dir, filename))

        if not json_files:
            return None

        all_rows = []
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            for result in data.get("results", []):
                row = {
                    "experiment_id": experiment_id,
                    "phase": "mid",
                    "eval_type": eval_type,
                    "epoch": data.get("epoch"),
                    "step": data.get("step"),
                    "scale": data.get("scale"),
                    "temperature": result.get("temp"),
                    "answer": result.get("answer"),
                    "probability": result.get("prob"),
                }
                all_rows.append(row)

        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_csv(mid_file, index=False)

            for json_file in json_files:
                os.remove(json_file)

            return mid_file
        return None

    @staticmethod
    def get_evaluation_inventory(dataset_name: str) -> str:
        """Return the inventory family appropriate for the dataset name."""
        if dataset_name.lower() == "pandora":
            return "personality"
        elif dataset_name.lower() == "emotion":
            return "emotion"
        else:
            return "unknown"
