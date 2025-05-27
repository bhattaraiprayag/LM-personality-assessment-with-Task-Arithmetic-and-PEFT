# run_baseline_eval.py

import pandas as pd
from src.eval_manager import EvalManager

def main():
    # Instantiate the evaluator for GPT-2
    evaluator = EvalManager(model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu")

    # List of inventories to evaluate
    inventories = ["BFI10", "PANASX", "IPIP120"]
    all_results = []

    # Run evaluations
    for inventory_name in inventories:
        print(f"Evaluating inventory: {inventory_name}")
        df = evaluator.score_likert(inventory_name=inventory_name)
        df.insert(0, "inventory", inventory_name)  # Add a column for inventory name
        all_results.append(df)

    # Combine and save
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("baseline_pre_trained.csv", index=False)
    print("Saved results to baseline_pre_trained.csv")

if __name__ == "__main__":
    import torch  # Ensure torch is imported here
    main()
