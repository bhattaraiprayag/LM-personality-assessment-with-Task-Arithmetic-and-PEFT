# Quickstart (uv)

## 1. Install `uv`

Follow the official install instructions: <https://docs.astral.sh/uv/getting-started/installation/>.

## 2. Sync the environment

From the repository root:

```bash
uv sync --dev
```

This installs runtime dependencies from `pyproject.toml` plus development tooling (including `ruff` and `mypy`).

## 3. Run code quality checks

```bash
uv run ruff format .
uv run ruff check . --fix
uv run mypy src scripts experiment_config.py
```

## 4. Run syntax compilation (no program execution)

```bash
uv run python -m compileall src scripts experiment_config.py
```

## 5. Start an experiment

```bash
uv run python -m scripts.start_experiment \
  --dataset pandora \
  --split agreeableness-bot-10 \
  --output outputs/ \
  --model_name gpt2 \
  --seed 183 \
  --epochs 3 \
  --batch_size 16 \
  --grad_steps 16
```

## 6. Example post-processing utilities

```bash
uv run python -m scripts.ft_results_processor
uv run python -m scripts.generate_ft_viz --results outputs_best/merged_results_scored.csv
```
