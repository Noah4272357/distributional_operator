# Gaussian Distributional Operator

This project learns an operator from a sampled input probability distribution
to a Gaussian output distribution. Given an unordered set of input particles,
the main model predicts the output mean and a positive-definite full covariance
matrix. Training uses the Gaussian negative log likelihood of observed output
particles; analytic target moments are reserved for validation metrics.

The repository also includes a moment-based neural baseline and a
distribution-space kernel-regression baseline.

## Problem and data

Each dataset item contains four arrays:

| Field | Shape | Meaning |
| --- | --- | --- |
| `input_dist` | `[data_size, sample_size, 4]` | Samples from an input Gaussian mixture |
| `output_dist` | `[data_size, sample_size, 4]` | Samples from the corresponding Gaussian output law |
| `target_mean` | `[data_size, 4]` | Analytic output mean |
| `target_cov` | `[data_size, 4, 4]` | Analytic output covariance |

The standard experiment uses 1,200 laws and 200 particles per law. The first
1,000 items after a deterministic permutation form the training split; the
remaining 200 items are used for validation and testing. The generated HDF5
dataset is intentionally ignored by version control.

Input laws are one-to-three-component diagonal Gaussian mixtures. Analytic
mixture moments and Fourier features are transformed by a fixed nonlinear map
to create the target Gaussian mean and covariance. See `data/plan.md` and
`data/generation.py` for the data contract and exact construction.

## Models

### Distributional operator

`distributional_operator` is the primary model. A shared particle encoder maps
each four-dimensional sample to a 64-dimensional embedding. Mean pooling makes
the representation permutation invariant. A second network maps the pooled
embedding to 128 features, and a Gaussian head predicts four mean entries and
the ten entries of a lower-triangular Cholesky factor.

The default architecture uses:

- two particle-encoder hidden layers of width 64;
- a 64-dimensional pooled embedding;
- three post-pooling hidden layers of width 128;
- GELU activations;
- a full covariance parameterization; and
- positive Cholesky diagonals with numerical stabilization of `1e-5`.

Checkpoints created before the model rename remain loadable through a legacy
identifier, but all new configurations and results use
`distributional_operator`.

### Moment MLP

`momentmlp` computes the empirical mean and unbiased empirical covariance of
the input particles, flattens those moments, and sends them through an MLP and
the same Gaussian prediction head.

### Kernel regression

`kernel_regression` fits empirical Gaussian moments to the input and output
particles in the training set. Query laws are compared with the training laws
using Gaussian Wasserstein distance, converted to radial-basis weights, and
used to interpolate empirical output means and covariances. It never uses
`target_mean` or `target_cov` for interpolation.

## Environment

Python 3.11 or 3.12 is required. Create the project environment from the lock
file:

```bash
uv sync --frozen --all-groups
```

The locked environment includes the CUDA 12.1 build of the tensor runtime. To
use a different accelerator or a CPU-only runtime, adjust the source selection
in `pyproject.toml` and regenerate the lock file.

## Generate the dataset

Generate the standard dataset with:

```bash
uv run python data/main.py --data-size 1200 --sample-size 200
```

This writes `data/dataset.h5`. Dataset generation is deterministic with seed
0. A generated dataset should remain local or be stored as an external
artifact rather than committed.

To compare empirical output moments with the analytic targets:

```bash
uv run python scripts/compare_generated_moments.py
```

## Configuration

The principal configurations are:

- `configs/config.yaml`: default 300-epoch distributional-operator experiment;
- `configs/distributional_operator_config.yaml`: 1,000-epoch five-seed setup;
- `configs/mlp_config.yaml`: 1,000-epoch moment-MLP setup.

Important defaults include a batch size of 64, AdamW with learning rate
`1e-3` and weight decay `0.01`, and a reduce-on-plateau scheduler. Validation
runs every 10 epochs, and the best checkpoint is selected by validation NLL.
Configuration values can be overridden with dot-list arguments.

## Train

Train the distributional operator:

```bash
uv run python scripts/train.py \
  --config configs/distributional_operator_config.yaml \
  --run-name distributional_operator_seed0
```

Run another seed by appending an override:

```bash
uv run python scripts/train.py \
  --config configs/distributional_operator_config.yaml \
  --run-name distributional_operator_seed1 \
  experiment.seed=1
```

The five-seed launcher runs the distributional operator and moment MLP with
seeds 0 through 4 and then produces an aggregate summary:

```bash
bash scripts/run_remote_seed_sweep.sh
```

## Evaluate and visualize

Evaluate a saved checkpoint on the held-out split:

```bash
uv run python scripts/evaluate.py \
  --checkpoint experiments/<run>/model_ckpt/best.pt \
  --split test
```

Generate publication-oriented mean and covariance figures:

```bash
uv run python scripts/plot_mean.py \
  --checkpoint experiments/<run>/model_ckpt/best.pt

uv run python scripts/plot_covariance.py \
  --checkpoint experiments/<run>/model_ckpt/best.pt
```

Both plotting tools write vector PDF and 400-dpi PNG outputs beside the run.

Validation reports the following metrics:

- output-particle Gaussian negative log likelihood (`nll`);
- excess NLL relative to the data-generating target Gaussian (`nll_diff`);
- Gaussian 2-Wasserstein distance (`w2_distance`);
- target-to-prediction KL divergence (`kl_divergence`); and
- Gaussian Hellinger distance (`hellinger_distance`).

All metrics are lower-is-better.

## Run outputs

Each training run creates a directory under `experiments/` containing:

```text
experiments/<run>/
├── config.yaml
├── train.log
├── training_history.csv
├── model_ckpt/
│   ├── best.pt
│   └── last.pt
├── results.json
├── results.csv
├── eval_by_split.csv
└── best_metrics.json
```

The best checkpoint embeds the resolved configuration, model and optimizer
states, scheduler state, epoch, validation score, and random-number-generator
state. Run outputs are ignored by version control.

## Project layout

```text
gaussian_law_exp/
├── configs/                 # Reproducible experiment configurations
├── data/                    # Synthetic generator and local HDF5 dataset
├── scripts/                 # Training, evaluation, plotting, and sweep tools
├── src/
│   ├── data/                # Dataset validation, splitting, and loaders
│   ├── models/              # Models, Gaussian head, and model factory
│   ├── training/            # Loss factory and training coordination
│   └── utils/               # Metrics, checkpoints, config, seeds, and logging
├── pyproject.toml
└── uv.lock
```

## Reproducibility notes

- Dataset generation and train/test splitting both use seed 0 by default.
- Training seeds control initialization and minibatch ordering.
- The reference seed sweep uses seeds `0, 1, 2, 3, 4`.
- The 200 held-out laws serve as both validation and test data, so reported
  results are validation-selected rather than estimates from an untouched
  third split.
- The HDF5 dataset, environments, caches, logs, checkpoints, and generated
  experiment artifacts are excluded from version control.
