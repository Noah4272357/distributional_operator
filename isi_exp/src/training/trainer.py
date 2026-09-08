"""Training coordination, checkpoint selection, artifacts, and final evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.build_model import model_metadata
from src.training.train_epoch import train_one_epoch
from src.training.validate import evaluate_isi_model, evaluate_isi_nll
from src.utils.artifacts import (
    BEST_METRICS_ARTIFACT,
    EVAL_BY_SPLIT_ARTIFACT,
    PREDICTION_CURVES_PT,
    RESULT_ARTIFACT,
    RESULT_TABLE_ARTIFACT,
    TRAINING_HISTORY_CSV,
)
from src.utils.checkpoint import (
    CHECKPOINT_SELECTION_METRIC,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from src.utils.reporting import (
    _flatten_eval_rows,
    _prediction_curves,
    _write_csv,
    _write_json,
    _write_training_history_csv,
)


class Trainer:
    """Coordinate epochs while keeping domain components outside the engine."""

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        dataloaders: dict[str, DataLoader],
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        device: torch.device,
        run_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = run_dir
        self.logger = logger
        self.checkpoint_dir = run_dir / "model_ckpt"
        self.best_path = self.checkpoint_dir / "best.pt"
        self.last_path = self.checkpoint_dir / "last.pt"

    def fit(self, resume: str | Path | None = None) -> dict[str, Any]:
        model_name = str(self.cfg.model.name).lower()
        metadata = model_metadata(model_name)
        start_epoch = 1
        resumed_best = float("inf")
        if resume is not None:
            checkpoint = load_checkpoint(resume, map_location="cpu")
            restored_epoch, resumed_best = restore_checkpoint(
                checkpoint,
                self.model,
                self.device,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            start_epoch = restored_epoch + 1
            self.logger.info("resumed checkpoint=%s epoch=%d", resume, restored_epoch)

        epoch_evaluator = (
            evaluate_isi_nll
            if str(self.cfg.training.get("epoch_metrics", "full")).lower() == "nll_only"
            else evaluate_isi_model
        )
        initial_val = epoch_evaluator(self.model, self.dataloaders["val"], self.device)
        current_val = float(initial_val["observable_metrics"]["observation_nll"])
        best_val = min(resumed_best, current_val)
        best_epoch = start_epoch - 1
        best_val_metrics = initial_val
        save_checkpoint(
            self.best_path,
            self.model,
            best_epoch,
            best_val,
            self.cfg,
            self.optimizer,
            self.scheduler,
        )

        history: list[dict[str, float | int]] = []
        grad_clip = self.cfg.training.get("grad_clip")
        if int(self.cfg.training.epochs) > 0 and self.optimizer is None:
            raise ValueError("models without trainable parameters require training.epochs=0")
        for epoch in range(start_epoch, int(self.cfg.training.epochs) + 1):
            train_loss = train_one_epoch(
                self.model,
                self.dataloaders["train"],
                self.optimizer,
                self.device,
                grad_clip=float(grad_clip) if grad_clip is not None else None,
            )
            train_metrics = epoch_evaluator(self.model, self.dataloaders["train"], self.device)
            val_metrics = epoch_evaluator(self.model, self.dataloaders["val"], self.device)
            val_nll = float(val_metrics["observable_metrics"]["observation_nll"])
            history_row = {
                "epoch": epoch,
                "train_observation_nll": float(train_metrics["observable_metrics"]["observation_nll"]),
                "val_observation_nll": val_nll,
            }
            for metric_name in (
                "tail_bin_error_against_empirical",
                "hellinger_distance_against_empirical",
                "kl_divergence_against_empirical",
                "finite_renormalized_w2_against_empirical",
            ):
                if metric_name in val_metrics["observable_metrics"]:
                    history_row[f"val_{metric_name}"] = float(val_metrics["observable_metrics"][metric_name])
            history.append(history_row)
            self.logger.info(
                "epoch=%d train_loss=%.8f val_observation_nll=%.8f lr=%.6g",
                epoch,
                train_loss,
                val_nll,
                float(self.optimizer.param_groups[0]["lr"]),
            )
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_nll)
            elif self.scheduler is not None:
                self.scheduler.step()
            if val_nll <= best_val:
                best_epoch = epoch
                best_val = val_nll
                best_val_metrics = val_metrics
                save_checkpoint(
                    self.best_path,
                    self.model,
                    epoch,
                    best_val,
                    self.cfg,
                    self.optimizer,
                    self.scheduler,
                )
            if bool(self.cfg.training.get("save_last", True)):
                save_checkpoint(
                    self.last_path,
                    self.model,
                    epoch,
                    best_val,
                    self.cfg,
                    self.optimizer,
                    self.scheduler,
                )

        if not self.last_path.exists() and bool(self.cfg.training.get("save_last", True)):
            save_checkpoint(
                self.last_path,
                self.model,
                best_epoch,
                best_val,
                self.cfg,
                self.optimizer,
                self.scheduler,
            )
        restore_checkpoint(load_checkpoint(self.best_path), self.model, self.device)
        metrics_by_split = {
            split: evaluate_isi_model(self.model, loader, self.device)
            for split, loader in self.dataloaders.items()
        }
        best_val_metrics = metrics_by_split["val"]
        results = {
            "model_name": model_name,
            "result_role": metadata["result_role"],
            "main_result_eligible": bool(metadata["main_result_eligible"]),
            "primary_model": bool(metadata["primary_model"]),
            "fixed_feature_main_baseline": bool(metadata["fixed_feature_main_baseline"]),
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "best_epoch": int(best_epoch),
            "best_checkpoint_path": str(self.best_path),
            "metrics_by_split": metrics_by_split,
        }
        best_metrics = {
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "best_epoch": int(best_epoch),
            "val": best_val_metrics,
        }
        rows = _flatten_eval_rows(results)
        _write_json(self.run_dir / RESULT_ARTIFACT, results)
        _write_json(self.run_dir / BEST_METRICS_ARTIFACT, best_metrics)
        _write_csv(self.run_dir / RESULT_TABLE_ARTIFACT, rows)
        _write_csv(self.run_dir / EVAL_BY_SPLIT_ARTIFACT, rows)
        _write_training_history_csv(self.run_dir / TRAINING_HISTORY_CSV, history)
        prediction_payload = _prediction_curves(self.model, self.dataloaders, self.device)
        prediction_payload["model_name"] = model_name
        prediction_payload["checkpoint_path"] = str(self.best_path)
        torch.save(prediction_payload, self.run_dir / PREDICTION_CURVES_PT)
        self.logger.info("best_epoch=%d best_val_observation_nll=%.8f", best_epoch, best_val)
        return results
