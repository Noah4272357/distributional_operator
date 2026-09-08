from pathlib import Path

import torch

from src.models.deepsets import DeepSets
from src.models.kernel_regression import KernelRegression
from src.training.factory import build_optimizer, build_scheduler, step_scheduler
from src.training.pipeline import TRAINING_HISTORY_FIELDNAMES
from src.utils.config import load_config, save_config


def test_plain_yaml_config_and_deepsets_name(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "config.yaml", ["experiment.epochs=3"])
    assert cfg.model.name == "deepsets"
    assert cfg.experiment.epochs == 3
    assert cfg.metrics.sinkhorn.enabled is True
    assert "val_field_sinkhorn_distance" in TRAINING_HISTORY_FIELDNAMES
    assert "learning_rate" in TRAINING_HISTORY_FIELDNAMES
    assert cfg.optimizer.name == "adamw"
    assert cfg.optimizer.weight_decay == 0.01
    assert cfg.scheduler.name == "reduce_lr_on_plateau"
    assert cfg.scheduler.patience == 5
    save_config(cfg, tmp_path)
    assert (tmp_path / "config.yaml").exists()


def test_adamw_and_reduce_lr_on_plateau_factories() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(
        model,
        {"name": "adamw", "lr": 1.0e-3, "weight_decay": 0.01},
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1.0e-3
    assert optimizer.param_groups[0]["weight_decay"] == 0.01

    scheduler = build_scheduler(
        optimizer,
        {
            "name": "reduce_lr_on_plateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 0,
            "threshold": 1.0e-4,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 1.0e-6,
            "eps": 1.0e-8,
        },
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    step_scheduler(scheduler, 1.0)
    step_scheduler(scheduler, 2.0)
    assert optimizer.param_groups[0]["lr"] == 5.0e-4


def test_deepsets_prediction_shapes() -> None:
    model = DeepSets(3, 2, 4, 1, 5, 6, 1, "gelu")
    prediction = model({"context_particles": torch.randn(4, 7, 3)})
    assert prediction["pred_mean"].shape == (4, 2)
    assert prediction["pred_cov"].shape == (4, 2, 2)


def test_kernel_regression_weights_and_prediction_shapes() -> None:
    historical_inputs = torch.randn(5, 6, 3)
    historical_mean = torch.randn(5, 2)
    historical_covariance = torch.eye(2).repeat(5, 1, 1)
    model = KernelRegression(
        historical_inputs,
        historical_mean,
        historical_covariance,
        bandwidth=0.5,
        sinkhorn_blur=0.1,
        history_chunk_size=2,
    )
    prediction = model({"context_particles": torch.randn(4, 6, 3)})
    assert prediction["pred_mean"].shape == (4, 2)
    assert prediction["pred_cov"].shape == (4, 2, 2)
    assert prediction["sinkhorn_distances"].shape == (4, 5)
    torch.testing.assert_close(prediction["kernel_weights"].sum(dim=-1), torch.ones(4))
