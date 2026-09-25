#!/usr/bin/env python3
"""Plot covariance and feature histograms of PCA Y from random training fields.

Five dataset items means five empirical measures, not five individual paths.
Their trajectories are pooled for one covariance matrix and one histogram per
PCA coordinate. The PCA transform is loaded from the training-fitted state;
this script never fits or overwrites a PCA transform.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from duffing_dataset.preprocessing import prepare_duffing_pca, transform_pca


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/distributional_operator.json"),
        help="config defining the training split and saved PCA state",
    )
    parser.add_argument("--seed", type=int, default=0, help="field-selection seed")
    parser.add_argument("--fields", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=13)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_coefficients(
    config: dict, project_root: Path, seed: int, fields: int, pca_dim: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select only training fields and apply the saved Y projection directly."""
    data = config["data"]
    preprocessing = data.get("preprocessing", {})
    if not preprocessing.get("enabled", False):
        raise ValueError("the config must enable PCA preprocessing")
    if str(preprocessing.get("name", "pca")).lower() != "pca":
        raise ValueError("only PCA preprocessing is supported")
    if data["target_dataset"] != "Y":
        raise ValueError("target_dataset must be 'Y'")
    if int(preprocessing["target_dimension"]) != pca_dim:
        raise ValueError("--pca-dim must match the config's target_dimension")
    input_path = (project_root / data["input_path"]).resolve()
    target_path = (project_root / data["target_path"]).resolve()
    with h5py.File(target_path, "r") as handle:
        shape = handle[data["target_dataset"]].shape
    if len(shape) != 3:
        raise ValueError("Y must have shape (fields, samples, grid)")
    train_size = int(data["train_size"])
    validation_size = int(data["validation_size"])
    test_size = int(data.get("test_size", 0))
    if train_size < 1 or validation_size < 1 or test_size < 0:
        raise ValueError("invalid training/validation/test split sizes")
    if train_size + validation_size + test_size > shape[0]:
        raise ValueError("configured splits exceed the dataset size")
    if not 1 <= fields <= train_size:
        raise ValueError("--fields must be between 1 and train_size")
    # Identical split construction to scripts/train.py.
    train_indices = np.random.default_rng(int(config["seed"])).permutation(
        shape[0]
    )[:train_size]
    state_path = (project_root / preprocessing["state_path"].format(
        seed=int(config["seed"]), train_size=train_size,
        input_dim=int(preprocessing["input_dimension"]), target_dim=pca_dim,
    )).resolve()
    if not state_path.is_file():
        raise FileNotFoundError(
            f"training-fitted PCA state is missing: {state_path}; "
            "run training preprocessing first"
        )
    state = prepare_duffing_pca(
        input_path, target_path, state_path, train_indices,
        input_dataset=data["input_dataset"], target_dataset=data["target_dataset"],
        input_dimension=int(preprocessing["input_dimension"]),
        target_dimension=pca_dim, rebuild=False,
        normalize=bool(preprocessing.get("normalize", False)),
        normalization_epsilon=float(preprocessing.get("normalization_epsilon", 1e-8)),
    )
    # Sort for h5py's increasing-index requirement; selection is without replacement.
    selected = np.sort(np.random.default_rng(seed).choice(
        train_indices, size=fields, replace=False
    ))
    with h5py.File(target_path, "r") as handle:
        raw_y = np.asarray(handle["Y"][selected], dtype=np.float32)
    if not np.isfinite(raw_y).all():
        raise ValueError("selected Y data contain NaN or infinity")
    coefficients = transform_pca(
        raw_y, state["target_mean"], state["target_components"],
        state.get("target_coefficient_mean"), state.get("target_coefficient_scale"),
    )
    if not np.isfinite(coefficients).all():
        raise ValueError("PCA coefficients contain NaN or infinity")
    metadata = {
        "config_seed": int(config["seed"]), "selection_seed": seed,
        "selected_training_field_indices": selected.tolist(),
        "training_fields": train_size, "selected_fields": fields,
        "paths_per_field": int(shape[1]), "pooled_observations": fields * shape[1],
        "pca_dimension": pca_dim, "pca_state": str(state_path),
        "normalized": bool(state["normalization_enabled"]),
        "source_hdf5": str(target_path), "source_dataset": "Y",
        "covariance_ddof": 1, "aggregation": "pooled trajectories across fields",
    }
    return coefficients, selected, metadata


def save_figure(figure, output: Path, dpi: int) -> None:
    figure.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_features(values: np.ndarray, covariance: np.ndarray, output: Path,
                  bins: int, dpi: int, fields: int) -> None:
    dimension = values.shape[-1]
    labels = [f"PC {index + 1}" for index in range(dimension)]
    scope = f"{fields} training fields; {len(values)} pooled Y trajectories"
    with plt.rc_context({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        figure, axis = plt.subplots(figsize=(8.0, 7.0))
        # A zero-centered signed scale, not a correlation normalization.
        limit = max(float(np.abs(covariance).max()), np.finfo(float).eps)
        palette = LinearSegmentedColormap.from_list(
            "signed_covariance", ["#1F4E79", "#FFFFFF", "#D97706"]
        )
        image = axis.imshow(
            covariance, cmap=palette, norm=Normalize(-limit, limit),
            interpolation="nearest", aspect="equal",
        )
        axis.set_xticks(range(dimension), labels, rotation=45, ha="right")
        axis.set_yticks(range(dimension), labels)
        axis.set_xlabel("Y PCA feature")
        axis.set_ylabel("Y PCA feature")
        axis.set_title(f"Y feature covariance (ddof=1)\n{scope}", pad=12)
        figure.colorbar(image, ax=axis, shrink=0.8, label="Covariance (PCA units²)")
        figure.tight_layout()
        save_figure(figure, output / "y_pca_covariance", dpi)

        columns = 4
        rows = (dimension + columns - 1) // columns
        figure, axes = plt.subplots(
            rows, columns, figsize=(12.0, 2.5 * rows), sharey=True, squeeze=False
        )
        for index, axis in enumerate(axes.flat):
            if index >= dimension:
                axis.set_visible(False)
                continue
            axis.hist(
                values[:, index], bins=bins, color="#1F4E79",
                edgecolor="white", linewidth=0.4,
            )
            axis.set_title(labels[index])
            axis.set_xlabel("PCA coefficient")
            if index % columns == 0:
                axis.set_ylabel("Trajectory count")
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#D9DDE3", linewidth=0.5)
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(
            f"Y PCA feature distributions\n{scope}; independent feature x-scales",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(figure, output / "y_pca_histograms", dpi)


def main() -> None:
    args = parse_args()
    if min(args.fields, args.pca_dim, args.bins, args.dpi) < 1:
        raise ValueError("fields, pca-dim, bins, and dpi must be positive")
    root = Path(__file__).resolve().parent
    config = json.loads((root / args.config).resolve().read_text(encoding="utf-8"))
    coefficients, selected, metadata = load_coefficients(
        config, root, args.seed, args.fields, args.pca_dim
    )
    pooled = coefficients.reshape(-1, args.pca_dim).astype(np.float64)
    if pooled.shape[0] < 2:
        raise ValueError("at least two trajectories are required for covariance")
    centered = pooled - pooled.mean(axis=0)
    covariance = centered.T @ centered / (pooled.shape[0] - 1)
    if not np.allclose(covariance, np.atleast_2d(np.cov(pooled, rowvar=False))):
        raise AssertionError("covariance cross-check failed")
    output = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None
        else root / "artifacts" / f"pca_y_features_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=False)
    plot_features(pooled, covariance, output, args.bins, args.dpi, args.fields)
    np.savez_compressed(
        output / "y_pca_statistics.npz", coefficients=coefficients,
        covariance=covariance, selected_training_indices=selected,
        feature_mean=pooled.mean(axis=0), feature_std=pooled.std(axis=0, ddof=1),
    )
    metadata["histogram_bins"] = args.bins
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_directory": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
