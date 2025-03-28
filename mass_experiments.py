import itertools
import subprocess
import time

# ==== CONFIGURABLE SECTION ====
splits = ["anger", "joy", "sadness", "optimism"]
# splits = ["anger", "joy"]
seeds = [183, 1, 2, 3, 4, 5, 6]
# seeds = [42]
batch_sizes = [4]
grad_steps = [4]
use_peft_options = [None, "lora"]
subset_options = [None]  # Optional prototyping
# subset_options = [120]  # Optional prototyping
# ==============================

def run_experiment(params):
    split, seed, batch_size, grad_step, use_peft, subset = params
    base_command = [
        "python", "start_experiment.py",
        "--dataset", "emotion",
        "--split", split,
        "--output", "outputs/",
        "--model_name", "gpt2",
        "--seed", str(seed),
        "--epochs", "1",
        "--batch_size", str(batch_size),
        "--grad_steps", str(grad_step),
    ]

    if use_peft:
        base_command += ["--use_peft", use_peft]
    if subset:
        base_command += ["--subset", str(subset)]

    print(f"\n[RUNNING] split={split} | seed={seed} | batch={batch_size} | grad_steps={grad_step} | peft={use_peft} | subset={subset}")
    start_time = time.time()
    subprocess.run(base_command)
    elapsed = time.time() - start_time
    print(f"[FINISHED] Elapsed: {elapsed:.2f}s\n{'-' * 60}")

def main():
    combinations = list(itertools.product(
        splits,
        seeds,
        batch_sizes,
        grad_steps,
        use_peft_options,
        subset_options,
    ))

    print(f"Total combinations to run: {len(combinations)}\n{'=' * 60}")

    for combo in combinations:
        run_experiment(combo)

if __name__ == "__main__":
    main()
