# ISI law-to-law experiment

This is a standalone refactor of the former `experiments/isi_law_to_law.py`
and `experiments/generate_isi_lif_laws.py` pipeline. Domain behavior is
preserved, while configuration, data, models, training, evaluation,
checkpointing, logging, and reproducibility have separate responsibilities.

## Quick start

```bash
uv sync
uv run data/generate_isi_lif_laws.py \
  --data-size 24 --sample-size 16 --n-isi 16 \
  --output generated/isi_lif_laws_smoke.h5
uv run scripts/train.py
uv run scripts/evaluate.py --checkpoint experiments/<run>/model_ckpt/best.pt
```

Configuration overrides use OmegaConf dot-list syntax:

```bash
uv run scripts/train.py training.epochs=20
```

Each training run receives a unique directory under `experiments/` with the
resolved `config.yaml`, `train.log`, metrics, prediction curves, and
resumable `best.pt` and `last.pt` checkpoints.

Dataset generation is self-contained in `data/` and depends on PyTorch and h5py.
It writes one compact dataset; `src/data/dataloader.py` creates deterministic
train, validation, and test subsets. See [data/README.md](data/README.md).

## Kernel-regression comparison

`kernel_regression` is a nonparametric Nadaraya-Watson baseline. It uses the
same HDF5 dataset, deterministic split, target masses, and evaluation metrics
as `isi_context_deepsets`. Its input-law distance is the existing finite
piecewise-uniform W2 implementation.

```bash
uv run scripts/train.py --config configs/table3.yaml \
  --run-name isi_context_deepsets_seed0 experiment.seed=0
uv run scripts/train.py --config configs/kernel_regression.yaml \
  --run-name kernel_regression_seed0
```

The two `results.json` files have the same metric schema. `run.sh` runs five
DeepSets seeds and the deterministic kernel baseline, then writes their test
comparison to `experiments/table3_matrix/aggregate.json` and
`table3_rows.tex`.

The generated HDF5 dataset also includes normalized drive parameters and
moment/RFF input features, so the parametric and fixed-feature models use the
same splits and targets:

```bash
uv run scripts/train.py --config configs/table3.yaml \
  --run-name isi_param_mlp_seed0 model.name=isi_param_mlp
uv run scripts/train.py --config configs/table3.yaml \
  --run-name isi_feature_mlp_seed0 model.name=isi_feature_mlp
```
