"""Lightweight regression tests for the standalone ISI refactor."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import generate_isi_lif_laws  # noqa: E402
from scripts import isi_law_to_law  # noqa: E402
from src.models.build_model import build_model  # noqa: E402
from src.models.kernel_regression import ISIKernelRegression  # noqa: E402
from src.training.factory import build_optimizer, build_scheduler  # noqa: E402
from src.training.metrics import categorical_metrics_against_empirical  # noqa: E402
from src.utils.config import load_config
from src.data.dataloader import split_dataset_payload  # noqa: E402


def _train_payload() -> dict[str, object]:
    return {
        "bin_counts": torch.ones(3, 5),
        "input_particles": torch.ones(3, 6),
        "normalized_params": torch.ones(3, 2),
        "input_features": torch.ones(3, 18),
    }


def test_configs_load_and_preserve_selection_metric() -> None:
    cfg = load_config("configs/config.yaml")
    assert cfg.training.checkpoint_selection_metric == isi_law_to_law.CHECKPOINT_SELECTION_METRIC
    assert cfg.data.train_size + cfg.data.val_size + cfg.data.test_size == 24


def test_factory_builds_every_preserved_model() -> None:
    cfg = load_config("configs/config.yaml")
    batch = {
        "input_particles": torch.ones(3, 6),
        "normalized_params": torch.ones(3, 2),
        "input_features": torch.ones(3, 18),
    }
    for name in ("isi_context_deepsets", "isi_param_mlp", "isi_feature_mlp"):
        cfg.model.name = name
        prediction = build_model(cfg.model, _train_payload())(batch)
        assert prediction["logits"].shape == (3, 5)


def test_kernel_regression_uses_the_shared_prediction_api() -> None:
    train_payload = {
        "law_ids": torch.arange(6),
        "input_particles": torch.linspace(-1, 1, 48).reshape(6, 8),
        "bin_counts": torch.ones(6, 5),
        "empirical_bin_mass": torch.softmax(
            torch.arange(30, dtype=torch.float32).reshape(6, 5), -1
        ),
    }
    model = ISIKernelRegression(train_payload, bandwidth=0.2, input_bins=8, distance_chunk_size=2)
    prediction = model(
        {
            "law_id": torch.tensor([0, 1]),
            "input_particles": train_payload["input_particles"][:2],
        }
    )
    assert prediction["logits"].shape == (2, 5)
    assert torch.allclose(prediction["pred_bin_mass"].sum(-1), torch.ones(2))
    assert sum(parameter.numel() for parameter in model.parameters()) == 0


def test_compatibility_facades_expose_original_apis() -> None:
    assert callable(generate_isi_lif_laws.generate_isi_lif_dataset)
    assert callable(isi_law_to_law.run_isi_experiment)
    assert callable(isi_law_to_law.compute_isi_loss)
    legacy = OmegaConf.create(
        {
            "generated_data": {"file": "dataset.pt"},
            "model": {"name": "isi_global_constant"},
            "optimizer": {"name": "adam", "lr": 1.0e-3, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "experiment": {"seed": 7, "device": "cpu", "epochs": 3, "batch_size": 5},
        }
    )
    adapted = isi_law_to_law._adapt_legacy_config(legacy)
    assert adapted.training.epochs == 3
    assert adapted.data.batch_size == 5


def test_dataloader_owns_deterministic_splitting() -> None:
    payload = {
        "law_ids": torch.arange(24),
        "regime_labels": torch.arange(24) % 3,
        "params": torch.ones(24, 2),
        "normalized_params": torch.ones(24, 2),
        "input_particles": torch.ones(24, 16),
        "input_features": torch.ones(24, 18),
        "bin_counts": torch.ones(24, 5),
        "empirical_bin_mass": torch.full((24, 5), 0.2),
        "bin_edges": torch.arange(5),
    }
    first = split_dataset_payload(payload, train_size=12, val_size=6, test_size=6, seed=7)
    second = split_dataset_payload(payload, train_size=12, val_size=6, test_size=6, seed=7)
    assert [len(first[name]["law_ids"]) for name in ("train", "val", "test")] == [12, 6, 6]
    assert all(torch.equal(first[name]["law_ids"], second[name]["law_ids"]) for name in first)
    assert set(torch.cat([first[name]["law_ids"] for name in first]).tolist()) == set(range(24))


def test_dataloader_supports_train_validation_split_without_test() -> None:
    payload = {
        "law_ids": torch.arange(12),
        "regime_labels": torch.arange(12) % 3,
        "params": torch.ones(12, 2),
        "normalized_params": torch.ones(12, 2),
        "input_particles": torch.ones(12, 16),
        "input_features": torch.ones(12, 18),
        "bin_counts": torch.ones(12, 5),
        "empirical_bin_mass": torch.full((12, 5), 0.2),
        "bin_edges": torch.arange(5),
    }
    splits = split_dataset_payload(payload, train_size=10, val_size=2, test_size=0, seed=0)
    assert tuple(splits) == ("train", "val")
    assert len(splits["train"]["law_ids"]) == 10
    assert len(splits["val"]["law_ids"]) == 2


def test_cli_help() -> None:
    for script in (
        "data/generate_isi_lif_laws.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "pred_heatmap.py",
    ):
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()


def test_cosine_annealing_scheduler_reaches_eta_min() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3)
    scheduler = build_scheduler(
        optimizer,
        OmegaConf.create({"name": "cosineAnnealing", "t_max": 4, "eta_min": 1.0e-5}),
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)


def test_reference_optimizer_and_plateau_scheduler_config() -> None:
    cfg = load_config("configs/config.yaml")
    model = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "min"
    assert scheduler.factor == pytest.approx(0.5)
    assert scheduler.patience == 5
    assert scheduler.threshold == pytest.approx(1.0e-4)
    assert scheduler.threshold_mode == "rel"
    assert scheduler.cooldown == 0
    assert scheduler.min_lrs == pytest.approx([1.0e-6])
    assert scheduler.eps == pytest.approx(1.0e-8)


def test_validation_metrics_are_limited_to_requested_set() -> None:
    empirical = torch.tensor([[0.5, 0.25, 0.25]])
    predicted = torch.tensor([[0.25, 0.50, 0.25]])
    metrics = categorical_metrics_against_empirical(
        {
            "logits": predicted.log(),
            "pred_bin_mass": predicted,
            "pred_cdf": predicted.cumsum(dim=-1),
        },
        {
            "bin_counts": empirical * 100,
            "empirical_bin_mass": empirical,
            "bin_edges": torch.tensor([0.0, 1.0, 2.0]),
        },
    )

    assert set(metrics) == {
        "observation_nll",
        "tail_bin_error_against_empirical",
        "hellinger_distance_against_empirical",
        "kl_divergence_against_empirical",
        "finite_renormalized_w2_against_empirical",
    }
    expected_hellinger = torch.sqrt(
        0.5 * (predicted.sqrt() - empirical.sqrt()).square().sum()
    ).item()
    expected_kl = (empirical * (empirical / predicted).log()).sum().item()
    assert metrics["hellinger_distance_against_empirical"] == pytest.approx(expected_hellinger)
    assert metrics["kl_divergence_against_empirical"] == pytest.approx(expected_kl)
