# experiment_runner.py

import subprocess
import datetime
import itertools
import os
import sys
import pandas as pd
from types import SimpleNamespace

from src.utilities import Utilities

# CONFIGURATION
OUTPUT_DIR = "outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "experiment_runner_log.txt")
CSV_FILE = os.path.join(OUTPUT_DIR, "experiment_tracker.csv")
USE_SUBSET = True   # True if using a subset of the data, False otherwise

base_config = {
    "dataset": "pandora",
    "subset": 500 if USE_SUBSET else None,
    "output": "outputs/",
    "model_name": "GPT2",
    "seed": 183,
    "epochs": 1,
    "batch_size": 2,
    "lr": 1e-5,
    "grad_steps": 2,
    "use_peft": "lora",
    "warmup_ratio": 0.03,
    "num_workers": 8,
}

tracker_dtypes = {
    "id": str,
    "trait": str,
    "type": str,
    "size": "Int64",
    "scale_peft": float,
    "status": str,
    "time_taken": str,
}

traits = ['agreeableness']      # ['agreeableness', 'conscientiousness', 'extraversion', 'neuroticism', 'openness']
types = ['top', 'bot'] # ['top', 'bot']
sizes = [5] # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
scale_peft_values = [1.0, 2.0]   # [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Log experiment status
def log_status(message, output=None, error=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"\n{'===' * 40}\n[{timestamp}] {message}\n")
        if output:
            log_file.write(f"OUTPUTS:{output}\n")
        if error:
            log_file.write(f"ERRORS:{error}\n")
    print(message)

# Create or update the CSV file with experiments
def initialize_tracker():
    try:
        experiments = [
            {
                "id": None,
                "trait": trait,
                "type": type_,
                "size": size,
                "scale_peft": scale_peft,
                "status": "pending",
                "start": None,
                "end": None,
                "time_taken": None,
            } for trait, type_, size, scale_peft in itertools.product(traits, types, sizes, scale_peft_values)
        ]
        if not os.path.exists(CSV_FILE):
            pd.DataFrame(experiments).to_csv(CSV_FILE, index=False)
        else:
            existing_data = pd.read_csv(CSV_FILE, dtype={
                k: v for k, v in tracker_dtypes.items() if k not in ['start', 'end']
            }, parse_dates=['start', 'end'])
            combined_data = pd.DataFrame(experiments).merge(
                existing_data[['id', 'trait', 'type', 'size', 'scale_peft', 'status', 'start', 'end', 'time_taken']],
                on=['trait', 'type', 'size', 'scale_peft'],
                how='left',
                suffixes=('', '_existing')
            )
            combined_data["status"] = combined_data["status_existing"].fillna("pending")
            combined_data['id'] = combined_data['id'].fillna(combined_data['id_existing']).astype(str)
            combined_data['start'] = combined_data['start_existing']
            combined_data['end'] = combined_data['end_existing']
            combined_data['time_taken'] = combined_data['time_taken_existing']
            combined_data.drop(columns=["status_existing", "id_existing", "start_existing", "end_existing", "time_taken_existing"], inplace=True)
            combined_data['id'] = combined_data['id'].astype(str)
            combined_data['id'] = combined_data['id'].replace('nan', None)
            combined_data['size'] = combined_data['size'].astype('Int64')
            combined_data.to_csv(CSV_FILE, index=False)
    except Exception as e:
        log_status(f"Error initializing tracker: {e}")
        exit(1)

# Run pending experiments and update CSV status
def run_experiments():
    tracker = pd.read_csv(CSV_FILE, dtype={
                k: v for k, v in tracker_dtypes.items() if k not in ['start', 'end']
            }, parse_dates=['start', 'end'])
    for index, row in tracker.iterrows():
        trait = row["trait"]
        type_ = row["type"]
        size = row["size"]
        scale_peft = row["scale_peft"]
        experiment_args = base_config.copy()
        split = f"{trait}-{type_}-{size}"
        experiment_args["split"] = split
        experiment_args["scale_peft"] = scale_peft
        experiment_args_ns = SimpleNamespace(**experiment_args)
        id = Utilities.generate_experiment_id(experiment_args_ns)
        tracker.at[index, "id"] = id
        exp_out_folder = os.path.join(OUTPUT_DIR, id)
        if os.path.exists(exp_out_folder):
            if row['status'] != 'completed':
                tracker.at[index, "status"] = "completed"
                log_status(f"Experiment {id} already completed. Skipping!")
                tracker.to_csv(CSV_FILE, index=False)
            continue
        current_config = base_config.copy()
        current_config.update({
            "split": split,
            "scale_peft": scale_peft,
        })
        command = [
            # "python", "start_experiment.py",
            sys.executable, "start_experiment.py",
            "--dataset", current_config["dataset"],
            "--split", current_config["split"],
            "--output", current_config["output"],
            "--model_name", current_config["model_name"],
            "--seed", str(current_config["seed"]),
            "--epochs", str(current_config["epochs"]),
            "--batch_size", str(current_config["batch_size"]),
            "--lr", str(current_config["lr"]),
            "--grad_steps", str(current_config["grad_steps"]),
            "--use_peft", current_config["use_peft"],
            "--scale_peft", str(current_config["scale_peft"]),
        ]
        if USE_SUBSET and current_config["subset"] is not None:
            command.extend(["--subset", str(current_config["subset"])])
        log_status(f"Starting experiment: ID: {id} | Split {split} | scale_peft {scale_peft}")

        start_time = datetime.datetime.now()
        tracker.at[index, "start"] = start_time
        os.makedirs(exp_out_folder, exist_ok=True)
        with open(os.path.join(exp_out_folder, "stdout.log"), "w") as stdout_file, \
                open(os.path.join(exp_out_folder, "stderr.log"), "w") as stderr_file:
            result = subprocess.run(command, stdout=stdout_file, stderr=stderr_file, cwd=os.getcwd())
        end_time = datetime.datetime.now()
        tracker.at[index, "end"] = end_time
        tracker.at[index, "time_taken"] = str(end_time - start_time)
        if result.returncode == 0:
            log_status(f"Completed experiment: ID: {id} | Split: {split} | scale_peft: {scale_peft}")
            tracker.at[index, "status"] = "completed"
        else:
            log_status(f"Experiment failed: ID: {id} | Split: {split} | scale_peft: {scale_peft} | Error: {result.returncode}")
            tracker.at[index, "status"] = "failed"
        # try:
        #     # result = subprocess.run(command, capture_output=True, text=True, check=True)
        #     result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=os.getcwd())
        #     end_time = datetime.datetime.now()
        #     time_taken = end_time - start_time
        #     tracker.at[index, "end"] = end_time
        #     tracker.at[index, "time_taken"] = str(time_taken)
        #     log_status(
        #         f"Completed experiment: ID: {id} | Split: {split} | scale_peft: {scale_peft}",
        #         output=result.stdout
        #     )
        #     tracker.at[index, "status"] = "completed"
        # except subprocess.CalledProcessError as e:
        #     end_time = datetime.datetime.now()
        #     time_taken = end_time - start_time
        #     tracker.at[index, "end"] = end_time
        #     tracker.at[index, "time_taken"] = str(time_taken)
        #     log_status(
        #         f"Experiment failed: ID: {id} | Split: {split} | scale_peft: {scale_peft} | Error: {e}",
        #         output=e.stdout if e.stdout else None,
        #         error=e.stderr if e.stderr else None
        #     )
        #     tracker.at[index, "status"] = "failed"
        tracker.to_csv(CSV_FILE, index=False)

# Initialize the tracker and run experiments
initialize_tracker()
run_experiments()
