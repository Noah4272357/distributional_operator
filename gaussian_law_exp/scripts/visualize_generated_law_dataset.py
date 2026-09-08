from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from data.io import save_yaml
from data.synthetic_laws import gaussian_mixture_diag_moments
from utils.artifacts import (
    COMPONENT_COUNT_HIST_PNG,
    FEATURE_SUMMARY_PNG,
    INPUT_LAW_MOMENTS_PNG,
    LAW_FEATURE_PCA_PNG,
    QA_SUMMARY_YAML,
    SELECTED_INPUT_LAW_PARTICLES_PCA_PNG,
    SELECTED_LAW_DISTANCE_MATRIX_PNG,
    SELECTED_TARGET_LAW_PARTICLES_PCA_PNG,
    TARGET_COV_EIGS_PNG,
    TARGET_MEAN_HIST_PNG,
)
from utils.generated_data_qa import (
    build_qa_summary,
    load_generated_dataset,
    pca_2d,
    select_representative_laws,
    selected_law_distances,
    standardize_features,
)


SPLIT_COLORS = {"train": "#2f6fbb", "val": "#d9822b", "test": "#2f9e44"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write QA visualizations for a generated law dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Path to a generated dataset directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory for QA artifacts. Defaults to <dataset-dir>/qa.")
    parser.add_argument("--selected-count", type=int, default=3, help="Number of representative train laws to plot.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split used for selected-law plots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "qa"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_generated_dataset(dataset_dir)
    payloads = loaded["splits"]
    selected_payload = payloads[args.split]
    selected_indices = select_representative_laws(selected_payload, count=int(args.selected_count))
    standardized_features, _, _ = standardize_features(selected_payload["oracle_features"])
    pca_info = pca_2d(standardized_features)
    qa_summary = build_qa_summary(dataset_dir, payloads, selected_indices, pca_info, selected_split=args.split)
    save_yaml(output_dir / QA_SUMMARY_YAML, qa_summary)

    _plot_component_count_hist(payloads, output_dir / COMPONENT_COUNT_HIST_PNG)
    _plot_input_law_moments(payloads, output_dir / INPUT_LAW_MOMENTS_PNG)
    _plot_target_mean_hist(payloads, output_dir / TARGET_MEAN_HIST_PNG)
    _plot_target_cov_eigs(payloads, output_dir / TARGET_COV_EIGS_PNG)
    _plot_feature_summary(payloads, output_dir / FEATURE_SUMMARY_PNG)
    _plot_law_feature_pca(payloads, output_dir / LAW_FEATURE_PCA_PNG)
    _plot_selected_particles_pca(
        selected_payload,
        selected_indices,
        output_dir / SELECTED_INPUT_LAW_PARTICLES_PCA_PNG,
        particle_key="context_particles",
        mean_values=_input_law_means(selected_payload),
        title="Selected input laws: context particles PCA",
    )
    _plot_selected_particles_pca(
        selected_payload,
        selected_indices,
        output_dir / SELECTED_TARGET_LAW_PARTICLES_PCA_PNG,
        particle_key="output_particles",
        mean_values=selected_payload["target_mean"],
        title="Selected target laws: output particles PCA",
    )
    _plot_selected_law_distance_matrix(
        selected_payload,
        selected_indices,
        output_dir / SELECTED_LAW_DISTANCE_MATRIX_PNG,
    )

    selected_law_ids = qa_summary["selected_law_ids"]
    print(f"qa_dir: {output_dir.as_posix()}")
    print(f"selected_law_ids: {selected_law_ids}")
    print(f"checks: {qa_summary['checks']}")
    return 0


def _plot_component_count_hist(payloads: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = sorted({int(value) for payload in payloads.values() for value in payload["input_law"]["component_count"].tolist()})
    width = 0.22
    x = torch.arange(len(labels), dtype=torch.float32)
    for offset, (split, payload) in enumerate(payloads.items()):
        counts = payload["input_law"]["component_count"]
        heights = [int((counts == label).sum().item()) for label in labels]
        ax.bar((x + (offset - 1) * width).tolist(), heights, width=width, label=split, color=SPLIT_COLORS.get(split))
    ax.set_xticks(x.tolist(), [str(label) for label in labels])
    ax.set_xlabel("Component count")
    ax.set_ylabel("Law count")
    ax.set_title("Component count distribution")
    ax.legend()
    _save_figure(fig, path)


def _plot_input_law_moments(payloads: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split, payload in payloads.items():
        mean, diag_cov = gaussian_mixture_diag_moments(
            payload["input_law"]["weights"],
            payload["input_law"]["means"],
            payload["input_law"]["diag_variances"],
        )
        axes[0].hist(mean.flatten().detach().cpu().numpy(), bins=30, alpha=0.5, label=split, color=SPLIT_COLORS.get(split))
        axes[1].hist(
            diag_cov.flatten().detach().cpu().numpy(),
            bins=30,
            alpha=0.5,
            label=split,
            color=SPLIT_COLORS.get(split),
        )
    axes[0].set_title("Input law analytic means")
    axes[1].set_title("Input law analytic diag covariance")
    for ax in axes:
        ax.set_ylabel("Count")
        ax.legend()
    _save_figure(fig, path)


def _plot_target_mean_hist(payloads: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, payload in payloads.items():
        ax.hist(payload["target_mean"].flatten().detach().cpu().numpy(), bins=40, alpha=0.5, label=split, color=SPLIT_COLORS.get(split))
    ax.set_title("Target mean distribution")
    ax.set_xlabel("Target mean value")
    ax.set_ylabel("Count")
    ax.legend()
    _save_figure(fig, path)


def _plot_target_cov_eigs(payloads: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split, payload in payloads.items():
        eigvals = torch.linalg.eigvalsh(payload["target_cov"])
        min_eig = eigvals.amin(dim=-1)
        condition = eigvals.amax(dim=-1) / eigvals.amin(dim=-1).clamp_min(1.0e-12)
        axes[0].hist(min_eig.detach().cpu().numpy(), bins=30, alpha=0.5, label=split, color=SPLIT_COLORS.get(split))
        axes[1].hist(condition.detach().cpu().numpy(), bins=30, alpha=0.5, label=split, color=SPLIT_COLORS.get(split))
    axes[0].set_title("Target covariance min eigenvalue")
    axes[1].set_title("Target covariance condition number")
    for ax in axes:
        ax.set_ylabel("Count")
        ax.legend()
    _save_figure(fig, path)


def _plot_feature_summary(payloads: dict[str, dict], path: Path) -> None:
    split_names = list(payloads)
    means = torch.stack([payloads[split]["oracle_features"].mean(dim=0) for split in split_names])
    stds = torch.stack([payloads[split]["oracle_features"].std(dim=0, unbiased=False) for split in split_names])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, matrix, title in [(axes[0], means, "Oracle feature means"), (axes[1], stds, "Oracle feature stds")]:
        image = ax.imshow(matrix.detach().cpu().numpy(), aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(split_names)), split_names)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02)
    axes[1].set_xlabel("Feature dimension")
    _save_figure(fig, path)


def _plot_law_feature_pca(payloads: dict[str, dict], path: Path) -> None:
    split_features = [payload["oracle_features"] for payload in payloads.values()]
    all_features = torch.cat(split_features, dim=0)
    standardized, _, _ = standardize_features(all_features)
    pca_info = pca_2d(standardized)
    coordinates = pca_info["coordinates"]
    fig, ax = plt.subplots(figsize=(7, 5))
    start = 0
    for split, payload in payloads.items():
        stop = start + int(payload["oracle_features"].shape[0])
        split_coordinates = coordinates[start:stop]
        ax.scatter(
            split_coordinates[:, 0].detach().cpu().numpy(),
            split_coordinates[:, 1].detach().cpu().numpy(),
            s=14,
            alpha=0.7,
            label=split,
            color=SPLIT_COLORS.get(split),
        )
        start = stop
    ratios = pca_info["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({ratios[0]:.2%})")
    ax.set_ylabel(f"PC2 ({ratios[1]:.2%})")
    ax.set_title("Law oracle-feature PCA")
    ax.legend()
    _save_figure(fig, path)


def _plot_selected_particles_pca(
    payload: dict,
    selected_indices: list[int],
    path: Path,
    *,
    particle_key: str,
    mean_values: torch.Tensor,
    title: str,
) -> None:
    selected_particles = payload[particle_key][selected_indices]
    law_ids = payload["law_ids"][selected_indices].tolist()
    particle_rows = selected_particles.reshape(-1, selected_particles.shape[-1])
    pca_info = pca_2d(particle_rows)
    coordinates = pca_info["coordinates"].reshape(len(selected_indices), selected_particles.shape[1], 2)
    projected_means = _project_with_pca(mean_values[selected_indices], pca_info)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2f6fbb", "#d9822b", "#2f9e44", "#9b59b6", "#c0392b"]
    for position, law_id in enumerate(law_ids):
        color = colors[position % len(colors)]
        ax.scatter(
            coordinates[position, :, 0].detach().cpu().numpy(),
            coordinates[position, :, 1].detach().cpu().numpy(),
            s=20,
            alpha=0.75,
            label=f"law {int(law_id)}",
            color=color,
        )
        ax.scatter(
            [float(projected_means[position, 0].item())],
            [float(projected_means[position, 1].item())],
            marker="x",
            s=80,
            color=color,
            linewidths=2,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    _save_figure(fig, path)


def _plot_selected_law_distance_matrix(payload: dict, selected_indices: list[int], path: Path) -> None:
    distances = selected_law_distances(payload, selected_indices)
    law_ids = [int(value) for value in payload["law_ids"][selected_indices].tolist()]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = {
        "oracle_feature": "Oracle feature",
        "target_mean": "Target mean",
        "target_covariance_frobenius": "Target covariance",
    }
    for ax, key in zip(axes, titles, strict=True):
        matrix = torch.tensor(distances[key], dtype=torch.float32)
        image = ax.imshow(matrix.numpy(), interpolation="nearest")
        ax.set_title(titles[key])
        ax.set_xticks(range(len(law_ids)), [str(value) for value in law_ids], rotation=45)
        ax.set_yticks(range(len(law_ids)), [str(value) for value in law_ids])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Selected-law distance matrices")
    _save_figure(fig, path)


def _input_law_means(payload: dict) -> torch.Tensor:
    mean, _ = gaussian_mixture_diag_moments(
        payload["input_law"]["weights"],
        payload["input_law"]["means"],
        payload["input_law"]["diag_variances"],
    )
    return mean


def _project_with_pca(values: torch.Tensor, pca_info: dict) -> torch.Tensor:
    return (values.to(dtype=torch.float32) - pca_info["center"]) @ pca_info["components"].transpose(0, 1)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
