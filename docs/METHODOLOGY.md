# Methodology

## Research setup

This project evaluates personality shifts in GPT-style causal language models under two adaptation modes:

1. **Baseline fine-tuning** (full model training).
2. **PEFT fine-tuning** with **LoRA** and scale sweeps for task arithmetic style analysis.

## Data sources and inventories

Primary data and scoring flows include:

- personality-conditioned splits derived from Pandora-style author/profile/comment data,
- emotion-conditioned training splits for affective transfer analysis,
- inventory-based evaluation with:
  - **BFI-10** (Big Five short form),
  - **PANAS-X** (affective profile),
  - **IPIP-120** (trait and facet level scoring).

Inventory prompts, anchors, and metadata are configured in `experiment_config.py`.

## Processing and training pipeline

Core pipeline modules:

- `src/data_preprocessor.py`: text cleaning and split preparation utilities.
- `src/data_manager.py`: dataset handling, tokenization, and dataloader construction.
- `src/model_manager.py`: training module and optimizer/scheduler setup.
- `src/peft_manager.py`: LoRA configuration and model wrapping.
- `src/eval_manager.py` and `src/eval_results_manager.py`: psychometric scoring and result persistence.
- `src/utils/pipeline.py` and `src/utils/main.py`: experiment orchestration.

## Experiment orchestration

Entrypoints and utility scripts live in `scripts/`:

- `scripts/start_experiment.py`: main training and evaluation run.
- `scripts/mass_exp_personality.py` and `scripts/mass_exp_emotion.py`: batch run orchestration.
- `scripts/ta_arim_exps.py`: adapter-scale sweep experiments.
- `scripts/ft_results_processor.py` and `scripts/ta_results_scorer.py`: scoring and aggregation.
- plotting scripts (`scripts/generate_ft_viz.py`, `scripts/h3_plots.py`, `scripts/plot_pandora_scores_heatmap_results.py`) for analysis artifacts.

## Validation strategy

Validation focuses on static and structural correctness without running experiment logic:

1. code formatting/linting with `ruff`,
2. static type checks with `mypy`,
3. syntax compilation via `python -m compileall`.
