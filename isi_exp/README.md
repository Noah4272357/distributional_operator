# ISI distribution-learning experiments

The active training code preserves three model pipelines:

- `distribution_operator`: predicts the ISI law from PCA-compressed stochastic-process paths;
- `isi_feature_mlp`: predicts the ISI law from fixed moment and Fourier features; and
- `kernel_regression`: a nonparametric Nadaraya--Watson reference based on empirical input laws.

## Environment and data

Python 3.9–3.12 and [`uv`](https://docs.astral.sh/uv/) are required. The lock
file selects the PyTorch 2.7.1 build for CUDA 12.6 used for the reported runs.

```bash
uv sync --frozen --all-groups
uv run data/generate_isi_lif_laws.py \
  --data-size 1200 --sample-size 200 --n-isi 200 \
  --output data/generated/isi_distribution_dataset.h5
uv run data/generate_isi_process_data.py \
  --source data/generated/isi_distribution_dataset.h5 \
  --output data/generated/isi_process_data.h5
```

The categorical target file contains the 49-bin ISI distributions, input
particles, and fixed features. The process file contains 200 trajectories on
256 time points for every law. See
[data/process_protocol.md](data/process_protocol.md) for the complete data and
preprocessing protocol.

## Train the retained models

```bash
# Process distribution operator
uv run scripts/train.py --config configs/process_train_1000.yaml

# Fixed-feature MLP
uv run scripts/train.py --config configs/feature_mlp.yaml

# Kernel regression
uv run scripts/train.py --config configs/kernel_regression.yaml
```

The default `configs/config.yaml` is a small feature-MLP smoke configuration.
Configuration overrides use OmegaConf dot-list syntax, for example:

```bash
uv run scripts/train.py --config configs/feature_mlp.yaml training.epochs=20
```

Each run creates a directory containing the resolved configuration, logs,
metrics, prediction curves, and checkpoints. Evaluate any retained model with:

```bash
uv run scripts/evaluate.py --checkpoint experiments/<run>/model_ckpt/best.pt
```

The process-operator five-seed launcher is:

```bash
bash scripts/run_5_seeds.sh
```

Create a publication heatmap from any retained model checkpoint with:

```bash
uv run scripts/pred_heatmap.py \
  --checkpoint experiments/<run>/model_ckpt/best.pt \
  --split val \
  --output experiments/<run>/pred_heatmap.pdf
```

Use `--split test` for the process distribution operator, whose current
configuration has a held-out test split rather than a validation split.

## Active scripts

The `scripts/` directory contains only entry points for running, evaluating,
or plotting the retained models:

```text
scripts/
  __init__.py
  train.py
  train.sh
  evaluate.py
  evaluate.sh
  run_5_seeds.sh
  pred_heatmap.py
```

`train.py` and `evaluate.py` support all three active model names through the
shared factory. `run_5_seeds.sh` runs the configured distribution operator for
seeds 0 through 4. `pred_heatmap.py` creates PDF and PNG comparisons of target
and predicted distributions.

## Active model layout

```text
src/models/
  build_model.py
  components.py
  distribution_operator.py
  feature_mlp.py
  kernel_regression.py
```

`build_model.py` accepts only `distribution_operator`, `isi_feature_mlp`, and
`kernel_regression`. Shared MLP and categorical-output utilities live in
`components.py`.

## Project layout

```text
isi_exp/
├── configs/       # Retained experiment configurations
├── data/          # Dataset generators and preprocessing protocol
├── scripts/       # Training, evaluation, plotting, and seed sweeps
├── src/
│   ├── data/      # Dataset loading and process preprocessing
│   ├── models/    # The three retained model pipelines
│   ├── training/  # Losses, metrics, validation, and training
│   └── utils/     # Checkpoints, configuration, logging, and reporting
├── tests/         # Standalone and data-generation tests
├── pyproject.toml
└── uv.lock
```

Generated datasets under `data/generated/` and run outputs under
`experiments/` are excluded from version control.
