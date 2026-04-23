"""Generate baseline evaluation results for a pre-trained model."""

import pandas as pd
import torch

from src.eval_manager import EvalManager


def main() -> None:
    """Evaluate the baseline model across all supported inventories."""
    evaluator = EvalManager(
        model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"
    )

    inventories = ["BFI10", "PANASX", "IPIP120"]
    all_results = []

    for inventory_name in inventories:
        print(f"Evaluating inventory: {inventory_name}")
        df = evaluator.score_likert(inventory_name=inventory_name)
        df.insert(0, "inventory", inventory_name)
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("baseline_pre_trained.csv", index=False)
    print("Saved results to baseline_pre_trained.csv")


if __name__ == "__main__":
    main()
