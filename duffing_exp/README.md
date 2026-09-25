# Duffing Distribution-Learning Experiments

This project learns the conditional distribution of Duffing-oscillator output
trajectories `Y` from empirical input trajectories `X`. The data are stored in
`duffing_dataset/duffing_dataset.h5` with shape
`(data_size, sample_size, grid_size)`.

## Retained pipelines

| Pipeline | Description | Configuration |
| --- | --- | --- |
| `momentMLP` | Per-sample MLP, DeepSets moment aggregation, conditional implicit MLP generator, and direct MLP decoder | `configs/momentMLP.json` |
| `distributional_operator` | Training-only PCA, DeepSets distribution encoder, conditional RealNVP, and inverse PCA reconstruction | `configs/distributional_operator.json` |
| `kernel_regression` | Nonparametric distributional kernel-regression baseline | `configs/kernel_regression.json` |

## Environment

Python 3.10–3.12 and [`uv`](https://docs.astral.sh/uv/) are required. The lock
file selects the PyTorch 2.6.0 build for CUDA 12.4 used for the reported runs.
Create the project-local environment with:

```bash
uv sync --frozen --all-groups
```

Run all commands below from `duffing_exp/`. The shell launchers use
`.venv/bin/python` by default; set `PYTHON_BIN` to override it.
The retained configurations default to CUDA; pass `--device cpu` to a training
or evaluation entry point for a CPU-only smoke run.

## Generate the dataset

Generate the standard 1,400-law dataset used by the checked-in configurations:

```bash
uv run python duffing_dataset/generate_dataset.py \
  --output duffing_dataset/duffing_dataset.h5 \
  --num-measures 1400 \
  --n-input 128 \
  --n-output 128
```

Dataset generation is computationally intensive. The generated HDF5 file is
excluded from version control. See
[`duffing_dataset/duffing_data_generation.md`](duffing_dataset/duffing_data_generation.md)
for the system, forcing process, numerical solver, and data protocol.

## Training

Train one run with an explicit result-directory name:

```bash
uv run python -m scripts.train \
  --config configs/momentMLP.json \
  --run-name momentMLP_seed0 \
  --seed 0

uv run python -m scripts.train \
  --config configs/distributional_operator.json \
  --run-name distributional_operator_seed0 \
  --seed 0
```

Run seeds `0,1,2,3,4` sequentially:

```bash
bash scripts/run_momentMLP_5_seeds.sh
bash scripts/run_distributional_operator_5_seeds.sh
```

Fit and evaluate the kernel-regression baseline:

```bash
uv run python scripts/train_kernel_regression.py \
  --config configs/kernel_regression.json
```

Each neural run writes its resolved `config.json`, metrics, and `best.pt` and
`last.pt` checkpoints beneath `experiments/<run_name>/`. PCA and normalization
statistics for `distributional_operator` are fitted only on the training split
and reused for validation and test data.

## Evaluation and plotting

Evaluate a saved checkpoint on the test split:

```bash
uv run python scripts/evaluate.py \
  --checkpoint experiments/distributional_operator_seed0/best.pt \
  --split test
```

Create the three-panel Duffing prediction figure:

```bash
uv run python scripts/plot_duffing.py \
  --checkpoint experiments/distributional_operator_seed4/best.pt \
  --output experiments/distributional_operator_seed4/duffing_test_indices_0_66_133.png
```

The default plotted test indices are `0`, `66`, and `133`.

## Supported scripts

Training and evaluation:

- `scripts/train.py` and `scripts/train.sh`: train `momentMLP` or
  `distributional_operator` from a config.
- `scripts/evaluate.py` and `scripts/evaluate.sh`: evaluate a neural checkpoint
  on its validation or test split.
- `scripts/train_kernel_regression.py`: fit and evaluate `kernel_regression`.
- `scripts/run_momentMLP_5_seeds.sh`: train `momentMLP` with seeds `0`–`4`.
- `scripts/run_distributional_operator_5_seeds.sh`: train
  `distributional_operator` with seeds `0`–`4`.
- `scripts/run_all_combinations.sh`: run both retained neural configurations.

Plotting:

- `scripts/plot_duffing.py`: plot three test measures from a retained neural
  checkpoint.
- `scripts/plot_mean_envlopes.py`: create the general mean and quantile-envelope
  comparison figure.
- `scripts/plot_pca_y_features.py`: plot target-PCA covariance and feature
  histograms using training-fitted preprocessing state.

## Source layout

- `src/models/`: retained model pipelines and shared components.
- `src/training/`: losses, metrics, validation, and trainer logic.
- `duffing_dataset/`: generation and training-only PCA preprocessing.
- `configs/`: the three supported experiment configurations.
- `scripts/`: supported training, evaluation, plotting, and multi-seed launchers.
- `experiments/`: checkpoints and metrics, grouped by run name.
- `pyproject.toml` and `uv.lock`: reproducible Python environment.

Generated datasets, experiment directories, checkpoints, caches, and local
environments are excluded from version control.
