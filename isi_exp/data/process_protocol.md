# Process-to-distribution data protocol

## Purpose

This protocol describes how to create the paired data used by the process
distribution operator. Each example is indexed by one parameter pair
\((m_i,q_i)\) and contains:

1. an input empirical law represented by 200 trajectories of a drifted
   Brownian process, sampled at 256 time points; and
2. a target interspike-interval (ISI) distribution from a leaky
   integrate-and-fire (LIF) first-passage model, represented by counts and
   probabilities in 48 finite-time bins plus one no-spike category.

The resulting process input has shape `(1200, 200, 256)`, and the categorical
target has shape `(1200, 49)`. The two files are paired by `law_ids`, not by an
assumed row order.

This document consolidates the design in
[`process_generation_plan.md`](../process_generation_plan.md), the model data
contract in [`process_model_plan.md`](../process_model_plan.md), and the LIF
generation details in [`README.md`](README.md). The values below describe the
current experiment.

## 1. Generate the source parameter laws and ISI targets

Create the source HDF5 file
`data/generated/isi_distribution_dataset.h5`. It contains the parameter laws,
the LIF-derived categorical targets, and auxiliary inputs used by other model
families.

### 1.1 Sample the parameter pairs

Generate 1,200 laws with master seed 0. Each law has parameters `(m, q)`, where
`m` is a drift parameter and `q` is a diffusion variance. Allocate 400 laws to
each of three regimes:

| Label | Regime | Distribution of `m` |
|---:|---|---|
| 0 | subthreshold | Uniform on `[0.75, 0.95]` |
| 1 | balanced near threshold | Uniform on `[0.95, 1.05]` |
| 2 | suprathreshold | Uniform on `[1.05, 1.25]` |

For all regimes, sample `q` independently from a log-uniform distribution on
`[0.1, 0.35]`. Equivalently,

\[
\log q \sim \operatorname{Uniform}(\log 0.1,\log 0.35).
\]

Parameter sampling uses seed `seed + 101`, which is 101 for the current
experiment. Assign unique integer `law_ids` from 0 through 1199 and retain the
regime label of each law.

### 1.2 Simulate the LIF first-passage targets

For each `(m, q)`, simulate 200 independent reset trials of

\[
dV_t = (-V_t/\gamma + m)\,dt + \sqrt{q}\,dW_t,
\]

with

\[
\gamma=1,\qquad V_0=0,\qquad V_{\mathrm{th}}=1.
\]

Use Euler--Maruyama with time step `0.002` over the observation window
`[0, 8]`. The random stream uses seed `seed + 303`, which is 303. When a step
crosses the threshold, linearly interpolate the crossing time between the
voltage values before and after the step. Mark a trial as censored when it does
not cross the threshold by time 8.

Divide `[0, 8]` into 48 equal-width bins with edges

\[
0,\frac{8}{48},\frac{2\cdot8}{48},\ldots,8.
\]

Store censored trials in a 49th no-spike category. For each law,
`bin_counts` contains the 49 category counts and `empirical_bin_mass` contains
the same counts divided by 200. Every probability row must sum to one.

### 1.3 Auxiliary arrays in the source file

The source generator also creates arrays used by the other model families:

- `input_particles`: 200 independent samples
  \(x_j=m+\sqrt{q}\epsilon_j\), with \(\epsilon_j\sim\mathcal N(0,1)\), using
  seed `seed + 202`;
- `normalized_params`: `(m, log(q))`; and
- `input_features`: the particle mean, population variance, eight empirical
  sine moments, and eight empirical cosine moments. The eight frequencies use
  seed `seed + 404` and Gaussian scale 1, yielding 18 features.

These auxiliary arrays remain in the target file for compatibility. The
process distribution operator uses `process_samples` as its input rather than
`input_particles` or `input_features`.

### 1.4 Source-generation settings

The complete source-generation settings are:

| Setting | Value |
|---|---:|
| Number of laws | 1200 |
| Input particles per law | 200 |
| LIF trials per law | 200 |
| Master seed | 0 |
| LIF time step | 0.002 |
| Observation horizon | 8.0 |
| Finite ISI bins | 48 |
| No-spike categories | 1 |
| Fourier frequencies | 8 |
| Fourier scale | 1.0 |

From the project root, the source dataset can be reproduced with:

```bash
.venv/bin/python data/generate_isi_lif_laws.py \
  --data-size 1200 \
  --sample-size 200 \
  --output generated/isi_distribution_dataset.h5 \
  --seed 0 \
  --device cuda \
  --n-isi 200 \
  --dt 0.002 \
  --t-max 8.0 \
  --finite-bins 48 \
  --feature-frequencies 8 \
  --rff-scale 1.0
```

The output uses the HDF5 format marker `isi_lif_laws`, format version 1, gzip
compression, and shuffled chunks.

## 2. Generate the matched process trajectories

Read `params`, `law_ids`, `regime_labels`, and `bin_edges` from
`data/generated/isi_distribution_dataset.h5`. Validate that `params` has shape
`(1200, 2)`, contains finite values, and has nonnegative `q`. Use the final
value of `bin_edges` as the process horizon, giving `T = 8`.

For every source law, generate 200 independent trajectories of

\[
dX_t=m\,dt+\sqrt{q}\,dW_t, \qquad X_0=0.
\]

Use a common grid of 256 points including both endpoints:

\[
t_k=k\Delta t,\qquad k=0,\ldots,255,\qquad
\Delta t=\frac{8}{255}.
\]

For path `j` of law `i`, draw independent increments

\[
\Delta W_{i,j,k}\sim\mathcal N(0,\Delta t),
\qquad k=1,\ldots,255,
\]

then compute

\[
W_{i,j,t_k}=\sum_{r=1}^{k}\Delta W_{i,j,r},
\qquad
X_{i,j,t_k}=m_i t_k+\sqrt{q_i}W_{i,j,t_k}.
\]

This is the exact Gaussian transition law on the requested grid because the
drift and diffusion coefficients are constant. It preserves temporal
correlation within a path; independently sampling each time marginal would
not produce valid trajectories.

### 2.1 Process randomization

Use master process seed 0. Derive a deterministic random stream for each law
with

```text
SeedSequence([master_seed, law_id]).
```

This makes each law reproducible independently of the generation batch size
or processing order. Within a law, all 200 paths and all 255 increments are
independent draws from that law's stream.

### 2.2 Process file and storage

Write `data/generated/isi_process_data.h5` with HDF5 format marker
`isi_drifted_brownian_paths`, format version 1. Its datasets are:

| Dataset | Shape | Dtype | Meaning |
|---|---:|---|---|
| `process_samples` | `(1200, 200, 256)` | `float32` | Process values ordered as law, path, time |
| `time_grid` | `(256,)` | `float32` | Shared grid from 0 through 8 |
| `params` | `(1200, 2)` | `float32` | Copied `(m, q)` values |
| `law_ids` | `(1200,)` | integer | Copied source identifiers |
| `regime_labels` | `(1200,)` | integer | Copied source regimes |

Generate eight laws at a time. Store `process_samples` with chunk shape
`(1, 200, 256)`, HDF5 shuffle filtering, and gzip compression level 4. Write
to a temporary file, validate it, set the `completed` attribute, and only then
rename it to the final path. The uncompressed process array contains
61,440,000 values and occupies approximately 246 MB; incremental writes avoid
holding the full array in memory during generation.

From the project root, reproduce the process file with:

```bash
.venv/bin/python data/generate_isi_process_data.py \
  --source data/generated/isi_distribution_dataset.h5 \
  --output data/generated/isi_process_data.h5 \
  --paths-per-law 200 \
  --time-points 256 \
  --seed 0 \
  --law-batch-size 8 \
  --compression-level 4
```

The generator records the equation, initial value, horizon, grid spacing,
path count, seed rule, source identity, storage settings, and validation
summary in the HDF5 metadata.

## 3. Validate the process data

Accept the process file only after the following checks pass.

### Structural and alignment checks

- `process_samples.shape == (1200, 200, 256)`;
- `time_grid.shape == (256,)`;
- all process values are finite;
- every trajectory starts at zero within absolute tolerance `1e-7`;
- the copied `params` and `law_ids` exactly match the source file; and
- the copied `regime_labels`, when present, match the source file.

### Grid checks

- the first and final grid values are 0 and 8;
- the grid is uniformly spaced at `8/255`; and
- adjacent spacing agrees with `8/255` using relative tolerance `1e-5` and
  absolute tolerance `5e-7`.

### Statistical checks

For each law, the exact moments are

\[
\mathbb E[X_t]=mt,\qquad \operatorname{Var}(X_t)=qt.
\]

Compare the empirical terminal mean and variance from 200 paths with `8m` and
`8q`. Also pool one-step increments within each law and compare their mean and
variance with `m(8/255)` and `q(8/255)`. The implemented acceptance limits are:

- maximum absolute terminal-mean z-score no greater than 6;
- terminal variance ratio between 0.5 and 1.7;
- maximum absolute increment-mean z-score no greater than 6; and
- increment variance ratio between 0.9 and 1.1.

These tolerances account for finite Monte Carlo sampling. Exact equality of
empirical and analytic moments is neither expected nor required.

## 4. Pair, split, and preprocess data for the model

The process file supplies the model inputs, while the source distribution file
supplies `bin_counts`, `empirical_bin_mass`, and `bin_edges`. Join the files by
unique `law_ids`. Require both files to contain exactly the same set of 1,200
IDs, and verify matched `params` and `regime_labels` before constructing any
split.

Use a deterministic regime-stratified split with split seed 0:

| Split | Laws | Individual process paths | Use |
|---|---:|---:|---|
| Training | 1000 | 200,000 | PCA fitting and model optimization |
| Test | 200 | 40,000 | Final evaluation only |

Keep all 200 trajectories associated with one law in the same split. The test
laws must not affect PCA fitting, truncation selection, optimization,
scheduling, early stopping, or checkpoint selection. This experiment has no
validation split; it trains for a fixed epoch count and evaluates the test set
after training.

### 4.1 Fit PCA using training trajectories only

Flatten the training process input as

```text
(1000, 200, 256) -> (200000, 256).
```

Compute the 256-coordinate training mean and covariance in float64 using
chunks. Center each time coordinate, but do not standardize coordinates by
their individual variances. Diagonalize the symmetric covariance matrix and
order its eigenvalues from largest to smallest.

Select the smallest integer `truncate_dim = d` satisfying the strict
criterion

\[
\frac{\sum_{r=1}^{d}\lambda_r}
     {\sum_{r=1}^{256}\lambda_r}>0.99.
\]

Use the resulting training mean and first `d` components to transform both
splits without refitting:

```text
training: (1000, 200, 256) -> (1000, 200, d)
test:     ( 200, 200, 256) -> ( 200, 200, d)
```

For the completed five-seed experiment, this procedure selected `d = 14` and
achieved a cumulative explained-variance ratio of approximately
`0.99015011`. The PCA state is identical across model seeds because the data
split and PCA fitting are controlled by split seed 0.

Store the mean, components, eigenvalues, explained-variance ratios, selected
dimension, threshold, split seed, and training law IDs in `pca_state.pt` and
inside every model checkpoint. Evaluation and plotting restore this state;
they do not fit PCA again.

## 5. Final model-facing data contract

After preprocessing, one batch contains:

| Field | Batch shape | Role |
|---|---:|---|
| `process_features` | `(B, 200, 14)` for the current data | PCA-compressed process trajectories |
| `bin_counts` | `(B, 49)` | Count-weighted training target |
| `empirical_bin_mass` | `(B, 49)` | Evaluation target distribution |
| `bin_edges` | `(49,)` | Edges of the 48 finite bins; the final category is no-spike |
| `law_id` | `(B,)` | Alignment and provenance identifier |
| `regime_label` | `(B,)` | Regime-specific evaluation grouping |
| `params` | `(B, 2)` | Original `(m, q)` metadata |

The model maps each compressed path through a shared two-layer MLP, aggregates
the 200 path features with a permutation-invariant DeepSets mean, and predicts
49 logits. Softmax converts these logits to `pred_bin_mass`, and cumulative
summation produces `pred_cdf`. Training uses the same count-weighted
multinomial negative log likelihood as the other ISI models.

## 6. Reproducibility checklist

Before using a regenerated dataset, confirm that:

1. both HDF5 files contain 1,200 unique and matching `law_ids`;
2. `(m, q)` and regime labels agree after joining by ID;
3. the target file has 49 count and probability categories per law;
4. the process file has shape `(1200, 200, 256)` and a completed validation
   marker;
5. the process horizon is inherited from the source and equals 8;
6. source seed offsets and the per-law process seed rule are recorded;
7. the 1,000/200 split uses split seed 0 and has no ID overlap;
8. PCA is fitted only on the 200,000 training trajectories;
9. the selected PCA dimension is the first dimension whose cumulative ratio
   is strictly greater than 0.99; and
10. the saved PCA state is restored unchanged for test evaluation and figure
    generation.
