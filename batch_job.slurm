#!/bin/bash
#SBATCH --job-name=pandora-finetuning
#SBATCH --partition=gpu_4           # choose your partition (you can check which ones are available with sinfo_t_idle) [options: gpu_4, gpu_8, gpu_4_a100, gpu_4_h100, dev_gpu_4, dev_gpu_4_a100]
#SBATCH --time=08:00:00  	        # computing time, after this, your job will be terminated
#SBATCH --nodes=1                   # compute nodes, for most jobs one is enough
#SBATCH --ntasks=1  		        # number of tasks across all nodes
#SBATCH --gres=gpu:3		        # claim 4 GPUs for your job (you should use more than one for e.g. model fine-tuning)
#SBATCH --output="pandora_out%j.log"	# writes the outputs of your job to a file "pandora_out"
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email when job fails
#SBATCH --mail-user=you@example.com                  # your email

conda deactivate
module del devel/cuda/11.8 jupyter/tensorflow/2023-10-10
source /path/to/venv/bin/activate
cd /path/to/LM-personality-assessment-with-Task-Arithmetic-and-PEFT/

# python batch_experiments.py
uv run python -m scripts.start_experiment --dataset pandora --split agreeableness-bot-10 --output outputs/ --model_name GPT2 --seed 183 --epochs 3 --batch_size 16 --grad_steps 16 --use_peft lora --scale_peft 0.5
