# Computation cost
The dominant costs are PCA preprocessing and repeated HDF5/checkpoint I/O—not GPU arithmetic.

Live status: PCA batch 17/33 after 27 minutes, using about 13 CPU cores and 3.3 GB RAM. Both GPUs remain idle until PCA finishes.

## 1. Dataset cost

Each tensor has shape:

```text
(500, 100, 101, 256)
```

Number of values per input or output:

```text
500 × 100 × 101 × 256 = 1,292,800,000
```

With `float32`:

- Input: 5.17 GB logical
- Output: 5.17 GB logical
- Total: 10.34 GB logical
- Compressed HDF5 size: 7.32 GB

Every epoch reads:

- Training data: approximately 8.27 GB logical
- Validation data: approximately 2.07 GB logical
- Total: 10.34 GB logical, or roughly 7.3 GB compressed

Across 300 epochs, that is approximately:

- 3.1 TB logical data processed
- Up to 2.2 TB read from the compressed HDF5 file

## 2. PCA preprocessing

The flattened field dimension is:

```text
101 × 256 = 25,856
```

Training PCA sees:

```text
400 fields × 100 samples = 40,000 observations
```

The current configuration fits up to 1,024 components for both input and output using 33 incremental batches.

Observed resource use:

- CPU: approximately 1,275%, or 12–13 cores
- Resident RAM: approximately 3.3 GB
- GPU: unused
- Measured projection: approximately 50–55 minutes total

Important risk: if 1,024 components do not explain at least 99% of both input and output variance, the program will stop after PCA and request a larger component limit.

## 3. Model size

Let \(p\) be the input PCA dimension. The feature width is:

\[
f=p^2.
\]

The model parameter count is dominated by:

\[
\text{parameters}\approx 385p^2.
\]

Approximate costs:

| PCA dimension | Feature width | Parameters | Adam checkpoint |
|---:|---:|---:|---:|
| 256 | 65,536 | 25 million | 0.3 GB |
| 512 | 262,144 | 101 million | 1.2 GB |
| 1,024 | 1,048,576 | 404 million | 4.8 GB |

“Adam checkpoint” includes model parameters and the two Adam moment tensors. Exact size also depends on the output PCA dimension.

Model size grows quadratically with the input PCA dimension. Doubling \(p\) makes the main network approximately four times larger.

## 4. GPU memory

At the worst-case \(p=1,024\):

- Parameters: approximately 1.6 GB
- Gradients: approximately 1.6 GB
- Adam states: approximately 3.2 GB
- PCA buffers: approximately 0.2 GB
- Activations and temporary tensors: likely 1–4 GB

Expected GPU usage is roughly 7–11 GB, so it should fit comfortably on the 24 GB RTX 4090.

The optimized expectation calculation in `model.py` is important here. Materializing `(batch,100,p²)` would require about 7.8 GB by itself at \(p=1,024\), whereas the implementation averages the smaller hidden representation before projecting to \(p²\).

## 5. Compute per epoch

For \(p=q=1,024\), one epoch requires approximately:

- 20 training batches
- 5 validation batches
- About 3–4 trillion floating-point operations
- About 10.3 GB of data decoding and transfer

The RTX 4090 can handle the arithmetic relatively quickly. HDF5 loading, decompression, CPU target-statistic calculation, and checkpoint writes will likely dominate wall time.

## 6. Checkpoint cost

The current code saves `last.pt` every epoch and overwrites `best.pt` whenever validation improves.

At \(p=1,024\):

- One checkpoint: approximately 4.8 GB
- `best.pt` + `last.pt`: approximately 9.6 GB
- Temporary atomic-save space: another approximately 4.8 GB
- Peak checkpoint storage: approximately 14–15 GB

The server currently has only about 30 GB free. It should fit, but the margin is limited.

Writing `last.pt` every epoch could produce approximately:

\[
4.8\text{ GB}\times300=1.44\text{ TB}
\]

of checkpoint write traffic. Frequent `best.pt` updates could push this toward 2–3 TB. This may add several hours and is the largest avoidable cost.

## 7. Expected wall time

Based on the live PCA timing and expected training I/O:

- PCA preprocessing: 50–55 minutes
- Training per epoch: provisionally 1–3 minutes
- 300 epochs: approximately 5–15 hours
- Checkpoint overhead may extend this further

A realistic total estimate is:

```text
6–18 hours
```

The estimate can be narrowed substantially after epoch 1 completes.

## Most effective optimizations

In order of likely impact:

1. Save `last.pt` every 10–20 epochs instead of every epoch.
2. Keep only `best.pt` plus occasional resumable checkpoints.
3. Cache PCA-transformed inputs instead of repeating the large PCA projection every epoch.
4. Precompute output mean and variance instead of loading all 100 output solutions every epoch.
5. Reduce the PCA threshold or dimension if scientifically acceptable.
6. Use both GPUs with distributed training only after addressing the I/O bottleneck.

The current job remains healthy at PCA batch 17/33 with no errors.

# Why sample to sample is a bad idea?
Suppose our purpose is to calculate the mean or variance of the output random field through sampling, then we need to sample around \[N\sim O(\frac{d}{\epsilon^2})\], which is around 10^6 for each random field, even for error like $(10^{-1})$.

The same reason explains why the stochastic heat equation task might be a bad idea, since the $\mathbb{E}[u(t,x)]$ can't be approximated well by the limited 100 samples.

# What tasks can be expected to solve?
Cost analysis: Input (N_rf, N_sample, Nx)

Target 1. Predict mean and variance of the output random field, then we need the exact expression, otherwise we might have to sample a lot of instances to get a good enough approximation.

Target 2. Output samples from random field.
