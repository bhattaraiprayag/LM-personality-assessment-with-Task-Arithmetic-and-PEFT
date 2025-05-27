import itertools
import subprocess
import time
import os
import json
import re
from statistics import mean
import shutil  # for detecting squeue


# # ==== CONFIGURATIONS ====
# # MAIN EXPERIMENTS
traits = ["agreeableness", "conscientiousness", "extraversion", "openness", "neuroticism"]
locations = ["top", "bot"]
sizes = [5, 10, 15]
seeds = [183, 1]
batch_sizes = [4]
grad_steps = [4]
use_peft_options = [None, "lora"]
subset_options = [2500]

# # # # PROTOTYPE EXPERIMENTS
# traits = ["agreeableness"]
# locations = ["top"]
# sizes = [5]
# seeds = [183]
# # grad_steps = [2]
# grad_steps = [4]
# # batch_sizes = [2]
# batch_sizes = [4]
# use_peft_options = [None, "lora"]
# # use_peft_options = ["lora"]
# # subset_options = [2500]
# subset_options = [100]

# ==============================


def is_cluster_environment():
    return shutil.which("squeue") is not None

TIME_STATS_PATH = os.path.join("outputs", "time_stats_personality.json")
TIME_BUFFER = 60  # 1 minute buffer to avoid overlap

if os.path.exists(TIME_STATS_PATH):
    with open(TIME_STATS_PATH, 'r') as f:
        time_stats = json.load(f)
else:
    time_stats = {"lora": {"times": []}, "none": {"times": []}}

def get_remaining_time():
    result = subprocess.run(["squeue", "-l"], stdout=subprocess.PIPE, text=True)
    match = re.search(r"RUNNING\s+(\d+):(\d+):(\d+)\s+(\d+):(\d+):(\d+)", result.stdout)
    if match:
        used_h, used_m, used_s, lim_h, lim_m, lim_s = map(int, match.groups())
        used_sec = used_h * 3600 + used_m * 60 + used_s
        lim_sec = lim_h * 3600 + lim_m * 60 + lim_s
        return lim_sec - used_sec
    else:
        return None

def get_time_estimate(use_peft):
    key = "lora" if use_peft else "none"
    times = time_stats[key]["times"]
    if not times:
        return 600  # default to 10 minutes if no prior data
    return max(mean(times), max(times))

def update_time_stat(use_peft, elapsed):
    key = "lora" if use_peft else "none"
    time_stats[key]["times"].append(elapsed)
    with open(TIME_STATS_PATH, 'w') as f:
        json.dump(time_stats, f, indent=2)


# Load metadata
METADATA_PATH = os.path.join("outputs", "experiment_metadata.json")
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'r') as f:
        completed_metadata = set(json.load(f).keys())
else:
    completed_metadata = set()

def generate_experiment_id_from_values(dataset, split, model_name, seed, epochs, use_peft, subset):
    dataset_abbr = dataset[0].upper() if dataset else "d"

    def abbreviate_split(split):
        if split:
            parts = split.split("-")
            if len(parts) >= 3:
                abbr = "".join([part[0] for part in parts[:2]]) + parts[2]
            else:
                abbr = "".join([part[0] for part in parts])
            return abbr.lower()
        else:
            return "split"

    split_abbr = abbreviate_split(split)
    model_abbr = model_name.lower() if model_name else "m"
    seed_str = f"Se-{seed}"
    epochs_str = f"Ep-{epochs}"
    id_parts = [dataset_abbr, split_abbr, model_abbr, seed_str, epochs_str]

    if use_peft is not None:
        id_parts.append(f"Pe-{use_peft}")
    if subset is not None:
        id_parts.append(f"Ss-{subset}")

    return "-".join(id_parts)

# Generate all split combinations: e.g., agreeableness-top-5
splits = [f"{trait}-{location}-{size}" for trait, location, size in itertools.product(traits, locations, sizes)]

def run_experiment(params, index, total):
    split, seed, batch_size, grad_step, use_peft, subset = params
    if is_cluster_environment():
        estimated_time = get_time_estimate(use_peft)
        print(f"[INFO] Estimated time for this experiment: {estimated_time:.2f}s")

        remaining = get_remaining_time()
        if remaining is None:
            print("[ERROR] Could not determine remaining time. Skipping experiment.")
            return
        if remaining < estimated_time + TIME_BUFFER:
            print(f"[SKIPPED] Not enough time left: {remaining}s vs needed {estimated_time + TIME_BUFFER}s")
            return
    else:
        print("[INFO] Running locally — skipping time estimate and squeue checks.")

    base_command = [
        "python", "start_experiment.py",
        "--dataset", "pandora",
        "--split", split,
        "--output", "outputs/",
        "--model_name", "gpt2",
        "--seed", str(seed),
        "--epochs", "3",
        # "--epochs", "1",  # For prototyping
        "--batch_size", str(batch_size),
        "--grad_steps", str(grad_step),
    ]

    if use_peft:
        base_command += ["--use_peft", use_peft]
    if subset:
        base_command += ["--subset", str(subset)]

    # os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\n[Experiment {index}/{total}]")
    print(f"[RUNNING] split={split} | seed={seed} | batch={batch_size} | grad_steps={grad_step} | peft={use_peft} | subset={subset}")
    start_time = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(base_command, env=env)
    # subprocess.run(base_command)
    elapsed = time.time() - start_time
    print(f"[FINISHED] Elapsed: {elapsed:.2f}s\n{'-' * 60}")

    if is_cluster_environment():
        update_time_stat(use_peft, elapsed)

    time.sleep(3)
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    combinations = list(itertools.product(
        splits,
        seeds,
        batch_sizes,
        grad_steps,
        use_peft_options,
        subset_options,
    ))

    total = len(combinations)
    print(f"Total combinations to run: {total}\n{'=' * 60}")

    to_run = []
    for combo in combinations:
        split, seed, batch_size, grad_step, use_peft, subset = combo
        exp_id = generate_experiment_id_from_values(
            dataset="pandora",
            split=split,
            model_name="gpt2",
            seed=seed,
            epochs=3,
            use_peft=use_peft,
            subset=subset
        )
        if exp_id not in completed_metadata:
            to_run.append(combo)
        else:
            print(f"[SKIPPED] {exp_id} already completed.")
    print(f"\nTOTAL new experiments to run: {len(to_run)}")

    for i, combo in enumerate(to_run, start=1):
        run_experiment(combo, i, len(to_run))

if __name__ == "__main__":
    main()
