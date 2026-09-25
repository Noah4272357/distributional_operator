# Standalone ISI LIF data generation

The `data/` folder contains the complete dataset generator. It does not read a
configuration file or import training code from `src/`. The generator creates
one unsplit HDF5 file; train, validation, and optional test splits are created
later by the data loader.

## Generate the current experiment dataset

Create an environment and install the two data-generation dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The dataset used by the current experiment is generated with:

```bash
.venv/bin/python generate_isi_lif_laws.py \
  --data-size 1200 \
  --sample-size 200 \
  --output generated/isi_lif_laws_n1200_s200_seed0.h5 \
  --seed 0 \
  --device cuda \
  --n-isi 200 \
  --dt 0.002 \
  --t-max 8.0 \
  --finite-bins 48 \
  --feature-frequencies 8 \
  --rff-scale 1.0
```

Except for the output filename and compute device, these values are the current
command-line defaults. CPU generation is supported by changing `--device` to
`cpu`. Both hyphenated and underscored forms of `data-size` and `sample-size`
are accepted. Relative output paths are resolved from `data/`, independent of
the shell's current working directory.

| Setting | Current value | Meaning |
|---|---:|---|
| `data_size` | 1200 | Number of independently parameterized input/output laws |
| `sample_size` | 200 | Input-law particles supplied for each law |
| `seed` | 0 | Master seed for all generated random streams |
| `n_isi` | 200 | Independent first-passage trials used for each target law |
| `dt` | 0.002 | Time step of the LIF simulation |
| `t_max` | 8.0 | End of the first-passage observation window |
| `finite_bins` | 48 | Equal-width finite ISI bins on `[0, 8]` |
| `feature_frequencies` | 8 | Frequencies used for fixed Fourier features |
| `rff_scale` | 1.0 | Standard deviation of the sampled frequencies |

With these settings, `input_particles` has shape `(1200, 200)`, while the
target count and probability arrays have shape `(1200, 49)`: 48 finite-time
bins plus one no-spike category.

## Data-generation method

### 1. Sample LIF drive parameters

Each law is defined by a drift parameter `m` and diffusion variance `q`. Laws
are allocated as evenly as possible among three regimes. For 1200 laws, each
regime contains 400 examples:

| Regime label | Name | Distribution of `m` |
|---:|---|---|
| 0 | subthreshold | uniform on `[0.75, 0.95]` |
| 1 | balanced-near-threshold | uniform on `[0.95, 1.05]` |
| 2 | suprathreshold | uniform on `[1.05, 1.25]` |

For every regime, `q` is sampled independently from a log-uniform distribution
on `[0.1, 0.35]`; equivalently, `log(q)` is uniform between `log(0.1)` and
`log(0.35)`.

### 2. Sample the empirical input law

For each parameter pair `(m, q)`, the generator draws `sample_size` scalar
particles

```text
x_j = m + sqrt(q) * epsilon_j,    epsilon_j ~ Normal(0, 1).
```

These particles form a finite empirical representation of the Gaussian input
law. They are the direct input to the kernel-regression baseline and the
source of the fixed features used by the feature MLP.

### 3. Simulate the target ISI law

The target law is generated with independent reset-trial simulations of the
leaky integrate-and-fire stochastic differential equation

```text
dV_t = (-V_t / gamma + m) dt + sqrt(q) dW_t,
```

using `gamma = 1`, initial voltage `V_0 = 0`, and threshold `V_th = 1`. The
simulation uses Euler--Maruyama updates with time step `dt`. When a step crosses
the threshold, the crossing time is linearly interpolated between the voltage
before and after that step. A trial that does not cross before `t_max` is marked
as censored and assigned to the no-spike category.

Under the current settings, each law uses 200 independent trials. Uncensored
crossing times are assigned to 48 equal-width bins with edges

```text
0, 8/48, 2*8/48, ..., 8.
```

The 49th category contains censored trials. `bin_counts` stores the resulting
counts, and `empirical_bin_mass` divides every row by its total count so that it
sums to one.

### 4. Construct auxiliary model inputs

`normalized_params` stores `(m, log(q))` for the parameter-based baseline.

`input_features` stores deterministic finite-sample summaries for the
fixed-feature baseline. The feature vector contains:

- the sample mean;
- the population variance;
- eight empirical sine moments; and
- eight empirical cosine moments.

For frequencies `omega_r`, the Fourier entries are

```text
mean_j sin(omega_r * x_j),    mean_j cos(omega_r * x_j).
```

The frequencies are drawn once from a zero-mean Gaussian distribution with
standard deviation `rff_scale`. Eight frequencies therefore produce
`2 + 2 * 8 = 18` input features.

## Reproducibility

The master seed is offset to create deterministic random streams for separate
generation stages:

| Stage | Generator seed |
|---|---:|
| Parameter sampling | `seed + 101` |
| Input-particle sampling | `seed + 202` |
| First-passage simulation | `seed + 303` |
| Fourier-frequency sampling | `seed + 404` |

Keeping the settings and compute device fixed reproduces the same generated
arrays. CPU and GPU generation use the same seed design but are not guaranteed
to be bitwise identical across different hardware or numerical backends.

## Output format and shapes

The output is one compressed HDF5 file with format marker `isi_lif_laws` and
format version `1`. It contains:

| Dataset | Shape with current settings | Meaning |
|---|---:|---|
| `law_ids` | `(1200,)` | Unique law identifiers |
| `regime_labels` | `(1200,)` | Integer drift-regime labels 0, 1, or 2 |
| `params` | `(1200, 2)` | Raw `(m, q)` parameters |
| `normalized_params` | `(1200, 2)` | `(m, log(q))` parameters |
| `input_particles` | `(1200, 200)` | Samples representing each input law |
| `input_features` | `(1200, 18)` | Moment and Fourier features |
| `bin_counts` | `(1200, 49)` | Target finite-bin and no-spike counts |
| `empirical_bin_mass` | `(1200, 49)` | Row-normalized target masses |
| `bin_edges` | `(49,)` | Edges of the 48 finite ISI bins |

The training data loader reads this unsplit file, applies a deterministic
regime-stratified shuffle, and slices all row-level arrays consistently. The
current experiment uses split seed 0, with 1000 training laws, 200 validation
laws, and no held-out test laws.

## Python API

The same dataset can be generated directly through the local Python API:

```python
from isi_generation import generate_isi_lif_dataset

result = generate_isi_lif_dataset(
    data_size=1200,
    sample_size=200,
    output_path="generated/isi_lif_laws_n1200_s200_seed0.h5",
    seed=0,
    device="cpu",
    n_isi=200,
    dt=0.002,
    t_max=8.0,
    finite_bins=48,
    feature_frequencies=8,
    rff_scale=1.0,
)
```

The returned dictionary contains the resolved dataset path and the generated
in-memory payload.

## Small CPU smoke dataset

For a fast end-to-end check:

```bash
.venv/bin/python generate_isi_lif_laws.py \
  --data-size 24 \
  --sample-size 16 \
  --n-isi 16 \
  --dt 0.004 \
  --t-max 6 \
  --finite-bins 16 \
  --output generated/isi_lif_laws_smoke.h5
```
