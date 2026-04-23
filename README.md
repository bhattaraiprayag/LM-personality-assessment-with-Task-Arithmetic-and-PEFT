# LM Personality Assessment with Task Arithmetic and PEFT

This is a research project investigating how adapter-based fine-tuning and task arithmetic affect language models' personality behavior/profiles.

## Documentation

- [docs/MOTIVATION.md](docs/MOTIVATION.md): project goals, research motivation, and expected outcomes.
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): datasets, modeling strategy, training/evaluation workflow, and architecture.
- [docs/QUICKSTART.md](docs/QUICKSTART.md): setup and command-line usage with `uv`.

## Repository layout

- `src/`: reusable training, evaluation, data, and PEFT modules.
- `scripts/`: experiment runners, scorers, and visualization pipelines.
- `experiment_config.py`: shared inventory and experiment constants.
- `batch_job.slurm`: SLURM job template.

## Scope

The codebase centers on:

1. dataset preparation for personality-conditioned splits,
2. GPT-style causal LM fine-tuning with and without LoRA,
3. inventory-based personality scoring (BFI-10, PANAS-X, IPIP-120),
4. post-processing and visualization for analysis of scale effects.
