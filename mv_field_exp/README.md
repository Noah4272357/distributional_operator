# McKean–Vlasov Field Experiment

This project generates and learns a single McKean–Vlasov Gaussian random-field law map. Generated tensors are stored together in one HDF5 file, while deterministic index arrays define the train, validation, and test subsets.

The default experiment generates:

- `data_size = 48` independent input laws;
- `sample_size = 16` input and output particles per law; and
- `grid_size = 32` spatial values per particle.

Thus the root HDF5 dataset `input_field_particles` has shape `(48, 16, 32)`. All three dimensions can be changed directly from the generation CLI.

## Installation

Run commands from the project root:

```bash
cd mv_field_exp
```

With `uv`:

```bash
uv sync --frozen
export PYTHON_BIN="$PWD/.venv/bin/python"
```

Or create a standard virtual environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install \
  "torch>=2.8,<2.9" "numpy>=1.26" "h5py>=3.11" "geomloss>=0.2.6,<0.3" \
  "matplotlib>=3.8" "pyyaml>=6" "pytest>=8"
export PYTHON_BIN="$PWD/.venv/bin/python"
```

For GPU training, use a PyTorch build compatible with the machine's CUDA runtime.

## Project structure

```text
mv_field_exp/
├── configs/
│   ├── generate.yaml                   # Generation defaults and equation parameters
│   └── config.yaml                     # Training, evaluation, and HDF5 paths
├── dataset/
│   ├── mckean_vlasov_generation.py     # Full generation and HDF5 assembly
│   └── io.py                           # HDF5 and YAML serialization
├── src/
│   ├── data/
│   │   ├── dataset.py                  # HDF5 split loading and law indexing
│   │   ├── preprocessing.py            # Deterministic standardization helpers
│   │   └── dataloader.py               # DataLoader construction
│   ├── models/
│   │   ├── mckean_vlasov.py             # Law sampling and closed-form target map
│   │   ├── bases.py                     # Sine basis and weighted PCA
│   │   ├── deepsets.py                  # Trainable empirical-law encoder
│   │   ├── kernel_regression.py         # Sinkhorn Nadaraya-Watson model
│   │   ├── baselines.py                 # Moment and global baselines
│   │   ├── gaussian_ops.py              # Gaussian algebra and metrics
│   │   └── build_model.py               # Model factory
│   ├── training/
│   │   ├── trainer.py                   # Training coordinator
│   │   ├── train_epoch.py               # Single-epoch interface
│   │   ├── validate.py                  # Validation interface
│   │   ├── factory.py                   # Optimizer factory
│   │   └── pipeline.py                  # Losses, metrics, checkpoints, loop
│   └── utils/                            # Config, logging, metrics, visualization
├── scripts/
│   ├── generate.py / generate.sh
│   ├── train.py / train.sh
│   ├── evaluate.py / evaluate.sh
│   ├── visualize_phase3_5_samples.py
│   └── visualize_random_field_covariance.py
├── experiments/                         # Timestamped training runs
├── tests/test_smoke.py
├── run.sh                               # Generate defaults, then train
└── pyproject.toml
```

The data flow is:

```text
configs/generate.yaml + CLI sizes
              │
              ▼
dataset/mckean_vlasov_generation.py
              │
              ├── src/models/mckean_vlasov.py
              └── src/models/bases.py
              │
              ▼
dataset/generated/McKean_Vlasov/McKean_Vlasov.h5
              │
              ▼
src/data/dataset.py → Trainer → best checkpoint and metrics
                                      │
                                      ├── evaluate.py
                                      └── visualization scripts
```

## Generate data

### Size arguments

The three primary sizes are explicit argparse options:

```bash
./scripts/generate.sh \
  --data_size 1000 \
  --sample_size 128 \
  --grid_size 64
```

This guarantees:

```text
input_field_particles.shape == (1000, 128, 64)
```

Hyphenated aliases are also accepted:

```bash
./scripts/generate.sh \
  --data-size 1000 \
  --sample-size 128 \
  --grid-size 64
```

When an option is omitted, its value comes from `configs/generate.yaml`:

```yaml
data_size: 48
sample_size: 16
grid_size: 32
```

The complete generation CLI is:

```text
--config PATH       generation YAML; default configs/generate.yaml
--data_size INT     total number of independent laws
--sample_size INT   input and output particles sampled for each law
--grid_size INT     number of physical-grid points in each field particle
overrides           optional plain-YAML dot-list assignments
```

Examples:

```bash
# Use all configured defaults.
./scripts/generate.sh

# Change sizes and the random seed.
./scripts/generate.sh \
  --data_size 256 \
  --sample_size 64 \
  --grid_size 128 \
  seed=7

# Change equation parameters and output filename.
./scripts/generate.sh \
  --data_size 500 \
  --sample_size 100 \
  --grid_size 64 \
  filename=McKean_Vlasov_t1.h5 \
  mckean_vlasov.T=1.0 \
  mckean_vlasov.kappa=0.02
```

`data_size` must be at least 3 because every HDF5 file includes nonempty train, validation, and test subsets. `sample_size` must be positive, `grid_size` must be at least 2, and `coefficient_rank` cannot exceed `grid_size`.

### Generation configuration

`configs/generate.yaml` controls:

| Key | Meaning |
|---|---|
| `seed` | Reproducibility seed. |
| `schema_version` | HDF5 data-contract version. |
| `output_dir`, `filename` | Destination of the single HDF5 file. |
| `data_size`, `sample_size`, `grid_size` | Defaults used when CLI flags are absent. |
| `coefficient_rank` | Number of sine coefficients defining each input Gaussian law. |
| `domain_min`, `domain_max` | Spatial grid interval. |
| `split.train_fraction`, `split.val_fraction` | Fractions used to create split index arrays; the remainder is test data. |
| `input_law.*` | Ranges for random Gaussian means, covariance scales, and SPD jitter. |
| `basis.*` | Train-only weighted-PCA rank selection. |
| `mckean_vlasov.kappa` | Diffusion coefficient. |
| `mckean_vlasov.a` | Linear fluctuation-decay coefficient. |
| `mckean_vlasov.b` | Mean-field coupling coefficient. |
| `mckean_vlasov.T` | Evolution time horizon. |

The coefficient-space map is Gaussian and closed form. For sine-mode Laplacian eigenvalues `lambda_k = (k pi)^2`, the input mean and covariance evolve as

```text
output_mean_k = input_mean_k exp(-(kappa lambda_k + a - b) T)

output_cov = D input_cov D,
D_kk = exp(-(kappa lambda_k + a) T).
```

## HDF5 format

Default output:

```text
dataset/generated/McKean_Vlasov/
├── McKean_Vlasov.h5
├── manifest.yaml
├── basis_metadata.yaml
├── projection_summary.yaml
└── generation_summary.yaml
```

All numerical data is stored at the root of `McKean_Vlasov.h5`. Let:

- `D = data_size`;
- `S = sample_size`;
- `G = grid_size`;
- `q = coefficient_rank`;
- `rin` and `rout` be the learned input/output PCA ranks.

| HDF5 dataset | Shape | Meaning |
|---|---:|---|
| `law_ids` | `(D,)` | Unique law identifiers. |
| `input_field_particles` | `(D, S, G)` | Input random-field samples on the physical grid. |
| `output_field_particles` | `(D, S, G)` | McKean–Vlasov output-field samples. |
| `input_projected_particles` | `(D, S, rin)` | Inputs in the train-fitted input PCA basis. |
| `output_projected_particles` | `(D, S, rout)` | Outputs in the train-fitted output PCA basis. |
| `input_law_mean` | `(D, q)` | Exact input Gaussian means in sine coefficients. |
| `input_law_cov` | `(D, q, q)` | Exact input Gaussian covariances. |
| `target_mean` | `(D, q)` | Exact output Gaussian means in sine coefficients. |
| `target_cov` | `(D, q, q)` | Exact output Gaussian covariances. |
| `target_scale_tril` | `(D, q, q)` | Cholesky factors of `target_cov`. |
| `target_output_grid_mean` | `(D, G)` | Exact output means on the grid. |
| `target_output_grid_cov` | `(D, G, G)` | Exact output covariance kernels on the grid. |
| `target_projected_output_mean` | `(D, rout)` | Exact output means in learned output coordinates. |
| `target_projected_output_cov` | `(D, rout, rout)` | Exact output covariances in learned output coordinates. |
| `split_indices/train` | variable | Row indices used for training. |
| `split_indices/val` | variable | Row indices used for checkpoint selection. |
| `split_indices/test` | variable | Held-out row indices. |

Root attributes record `schema_version`, `target_name`, `seed`, `data_size`, `sample_size`, `grid_size`, and `coefficient_rank`.

Inspect a file with:

```python
import h5py

path = "dataset/generated/McKean_Vlasov/McKean_Vlasov.h5"
with h5py.File(path, "r") as data:
    print(dict(data.attrs))
    print(data["input_field_particles"].shape)
    print(data["split_indices/train"][:])
```

The weighted-PCA bases are fitted using training rows only and then applied to every row. `basis_metadata.yaml` contains the physical grid, quadrature weights, centers, basis vectors, and explained variance required to decode model predictions.

## Train

Generate the HDF5 file first:

```bash
./scripts/generate.sh
./scripts/train.sh --config configs/config.yaml
```

Or use the combined default workflow:

```bash
./run.sh
```

Training overrides use plain YAML dot-list syntax:

```bash
# One-epoch CPU check.
./scripts/train.sh experiment.device=cpu experiment.epochs=1

# Smaller DeepSets model.
./scripts/train.sh \
  model.inner_width=16 \
  model.embedding_dim=16 \
  model.outer_width=32 \
  model.inner_layers=1 \
  model.outer_layers=1

# Fixed output path rather than an automatic timestamp.
./scripts/train.sh --run-dir experiments/manual_debug experiment.epochs=2
```

Training arguments:

| Argument | Meaning |
|---|---|
| `--config PATH` | Training configuration; defaults to `configs/config.yaml`. |
| `--run-dir PATH` | Exact output directory. Omit it for an automatic timestamped directory. |
| positional overrides | Plain-YAML assignments merged into the configuration. |

Important configuration choices:

- `experiment.device`: `cpu`, `cuda`, or a device such as `cuda:1`.
- `experiment.epochs`, `batch_size`, `num_workers`, `pin_memory`: loop and loader settings.
- `generated_data.file`: HDF5 input path.
- `random_field_view.input_representation`: `learned_basis_projected` or `direct_grid`.
- `random_field_view.context_particles`, `output_particles`: optional positive particle caps; `null` uses all particles.
- `standardization.enabled`: fit feature statistics on training data and reuse them for validation/test.
- `model.name`: `deepsets`, `kernel_regression`, `moment_only`, or `global_constant`.
- `model.activation`: `relu`, `gelu`, `tanh`, `silu`, or `identity`.
- `model.covariance_structure`: `full`, `diagonal`, or `low_rank_diagonal`.
- `loss.name`: `gaussian_nll`, `param_supervised`, `gaussian_nll_plus_param`, `field_marginal_ce`, `gaussian_nll_plus_field_marginal_ce`, `coefficient_energy`, or `gaussian_nll_plus_energy`.
- `optimizer.name`: `adamw` by default (`adam` remains available); configure it with `lr` and `weight_decay`.
- `scheduler.name`: `reduce_lr_on_plateau` lowers the learning rate when the validation checkpoint score stops improving. Its choices are `mode`, `factor`, `patience`, `threshold`, `threshold_mode`, `cooldown`, `min_lr`, and `eps`; use `none` to disable scheduling.
- `metrics.sinkhorn`: GeomLoss validation settings (`p`, `blur`, `scaling`, `debias`, and `backend`).
- `metrics.*`: predicted particle count and other field-level metric settings.

The default optimizer/scheduler settings mirror `remote_1200.yaml`: AdamW with learning rate `1e-3` and weight decay `0.01`, followed by `ReduceLROnPlateau` in minimization mode with factor `0.5`, patience `5`, relative threshold `1e-4`, and minimum learning rate `1e-6`. The scheduler monitors the same validation score used to select the best checkpoint, and `training_history.csv` records the learning rate used for every epoch.

The `kernel_regression` model implements distribution-space Nadaraya–Watson regression. For a query input law, it computes GeomLoss Sinkhorn distances to all training input laws, applies weights proportional to `exp(-distance² / (2 bandwidth²))`, and moment-matches the weighted historical output Gaussians. Configure it with:

```bash
./scripts/train.sh \
  model.name=kernel_regression \
  model.bandwidth=0.5 \
  model.sinkhorn.p=2 \
  model.sinkhorn.blur=0.05 \
  model.sinkhorn.scaling=0.9 \
  model.sinkhorn.debias=true \
  model.sinkhorn.backend=tensorized \
  model.history_chunk_size=64
```

Kernel regression is non-parametric: it does not run gradient epochs, so its selected checkpoint reports epoch 0. Its checkpoint stores the historical training laws and target moments as buffers. `history_chunk_size` limits how many historical laws are compared per Sinkhorn batch to control memory use. The model currently requires `standardization.enabled=false`.

If generation uses a different `output_dir` or `filename`, update `generated_data.path`, `generated_data.file`, and the four metadata paths under `generated_data.files` in `configs/config.yaml`.

Each run produces:

```text
experiments/<experiment.name>/<timestamp>/
├── config.yaml
├── train.log
└── artifacts/
    ├── model_ckpt/best.pt
    ├── results.json
    ├── results.csv
    ├── best_metrics.json
    ├── eval_by_split.csv
    ├── training_history.csv
    └── copies of generation/basis metadata
```

## Evaluate

Evaluation loads the saved config and best checkpoint without retraining:

```bash
RUN=experiments/McKean_Vlasov_deepsets_smoke/<timestamp>

./scripts/evaluate.sh \
  --run-dir "$RUN" \
  --split test \
  --device cuda
```

Options:

- `--run-dir`: required run directory.
- `--checkpoint`: optional checkpoint path; defaults to `artifacts/model_ckpt/best.pt`.
- `--split`: `train`, `val`, or `test`; default `test`.
- `--device`: evaluation device; defaults to the saved training device.

The command prints JSON containing the training objective, coefficient-space metrics, field-space metrics, and Gaussian diagnostics. Validation and evaluation include `field_sinkhorn_distance`, computed by GeomLoss between quadrature-weighted predicted and reference field-particle clouds when `metrics.sinkhorn.enabled=true`.

## Visualize and create figures

On a headless server, first use a writable Matplotlib cache:

```bash
export MPLCONFIGDIR=/tmp/mckean-vlasov-matplotlib
```

### True versus predicted fields

```bash
"$PYTHON_BIN" scripts/visualize_phase3_5_samples.py \
  --runs-root experiments \
  --output-dir figures/field_samples \
  --splits train test \
  --laws-per-split 4 \
  --predicted-samples 16 \
  --device cpu \
  --output-format pdf
```

This discovers completed runs, selects laws, samples predicted fields, and creates one true-versus-predicted panel per requested split.

### Covariance figures

```bash
RUN=experiments/McKean_Vlasov_deepsets_smoke/<timestamp>

"$PYTHON_BIN" scripts/visualize_random_field_covariance.py \
  --run-dir "$RUN" \
  --output-dir "$RUN/figures" \
  --split test \
  --device cpu \
  --output-format png \
  --figure-kind all
```

`--figure-kind` may be repeated. Choices are:

- `all`;
- `coefficient-summary`;
- `coefficient-cov-heatmap`;
- `field-cov-slice`;
- `coefficient-scatter`; and
- `field-scatter`.

By default the script selects the best, middle, and worst laws by coefficient-covariance error. Selection can instead use `--selection-score field-cov` or `combined`, or explicit IDs:

```bash
"$PYTHON_BIN" scripts/visualize_random_field_covariance.py \
  --run-dir "$RUN" \
  --output-dir "$RUN/figures_explicit" \
  --law-selection explicit \
  --law-ids 40 43 47 \
  --figure-kind coefficient-scatter \
  --coeff-pairs 0,1 0,2 \
  --output-format pdf
```

Other controls include `--best-k`, `--middle-k`, `--worst-k`, `--max-coefficients`, `--num-coeff-pairs`, `--field-slice-anchors`, `--field-pairs`, and `--ellipse-sigma`.

## Validate

```bash
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/generate.py --help
"$PYTHON_BIN" scripts/train.py --help
"$PYTHON_BIN" scripts/evaluate.py --help
"$PYTHON_BIN" scripts/visualize_random_field_covariance.py --help
```

The tests cover configuration loading, direct HDF5 generation, the exact root tensor shape, and the argparse size options.
