"""Training coordination, checkpoint selection, artifacts, and final evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass
class EarlyStopping:
    """Stop after consecutive validation epochs without a significant improvement."""

    enabled: bool = True
    patience: int = 30
    min_delta: float = 0.001
    best: float = float("inf")
    epochs_without_improvement: int = 0

    def update(self, value: float) -> bool:
        if not self.enabled:
            return False
        if value < self.best - self.min_delta:
            self.best = float(value)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def state_dict(self) -> dict[str, float | int]:
        return {
            "best": float(self.best),
            "epochs_without_improvement": int(self.epochs_without_improvement),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best = float(state.get("best", self.best))
        self.epochs_without_improvement = int(
            state.get("epochs_without_improvement", self.epochs_without_improvement)
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
        preprocessing_state: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = run_dir
        self.logger = logger
        self.preprocessing_state = preprocessing_state
        self.checkpoint_dir = run_dir / "model_ckpt"
        self.best_path = self.checkpoint_dir / "best.pt"
        self.last_path = self.checkpoint_dir / "last.pt"

    def fit(self, resume: str | Path | None = None) -> dict[str, Any]:
        if "val" not in self.dataloaders:
            return self._fit_without_validation(resume)
        model_name = str(self.cfg.model.name).lower()
        metadata = model_metadata(model_name)
        early_cfg = self.cfg.training.early_stopping
        early_stopping = EarlyStopping(
            enabled=bool(early_cfg.enabled),
            patience=int(early_cfg.patience),
            min_delta=float(early_cfg.min_delta),
        )
        start_epoch = 1
        resumed_best = float("inf")
        resume_checkpoint: dict[str, Any] | None = None
        if resume is not None:
            resume_checkpoint = load_checkpoint(resume, map_location="cpu")
            restored_epoch, resumed_best = restore_checkpoint(
                resume_checkpoint,
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
        if resume_checkpoint is None:
            best_val = current_val
            best_epoch = 0
            early_stopping.update(current_val)
        else:
            trainer_state = resume_checkpoint.get("trainer_state") or {}
            best_val = resumed_best
            best_epoch = int(trainer_state.get("best_epoch", resume_checkpoint.get("epoch", 0)))
            early_state = trainer_state.get("early_stopping")
            if isinstance(early_state, dict):
                early_stopping.load_state_dict(early_state)
            else:
                early_stopping.best = best_val
            if not self.best_path.exists() or current_val < best_val:
                best_val = current_val
                best_epoch = start_epoch - 1
        best_val_metrics = initial_val

        def trainer_state() -> dict[str, Any]:
            return {
                "best_epoch": int(best_epoch),
                "early_stopping": early_stopping.state_dict(),
            }

        if resume_checkpoint is None or not self.best_path.exists() or current_val < resumed_best:
            save_checkpoint(
                self.best_path,
                self.model,
                best_epoch,
                best_val,
                self.cfg,
                self.optimizer,
                self.scheduler,
                trainer_state=trainer_state(),
                preprocessing_state=self.preprocessing_state,
            )

        history: list[dict[str, float | int]] = []
        grad_clip = self.cfg.training.get("grad_clip")
        last_epoch = start_epoch - 1
        stopped_early = False
        if int(self.cfg.training.epochs) > 0 and self.optimizer is None:
            raise ValueError("models without trainable parameters require training.epochs=0")
        for epoch in range(start_epoch, int(self.cfg.training.epochs) + 1):
            last_epoch = epoch
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
            should_stop = early_stopping.update(val_nll)
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
                    trainer_state=trainer_state(),
                    preprocessing_state=self.preprocessing_state,
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
                    trainer_state=trainer_state(),
                    preprocessing_state=self.preprocessing_state,
                )
            if should_stop:
                stopped_early = True
                self.logger.info(
                    "early_stopping epoch=%d patience=%d min_delta=%.8g reference_val=%.8f",
                    epoch,
                    early_stopping.patience,
                    early_stopping.min_delta,
                    early_stopping.best,
                )
                break

        if not self.last_path.exists() and bool(self.cfg.training.get("save_last", True)):
            save_checkpoint(
                self.last_path,
                self.model,
                best_epoch,
                best_val,
                self.cfg,
                self.optimizer,
                self.scheduler,
                trainer_state=trainer_state(),
                preprocessing_state=self.preprocessing_state,
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
            "last_epoch": int(last_epoch),
            "stopped_early": bool(stopped_early),
            "best_checkpoint_path": str(self.best_path),
            "metrics_by_split": metrics_by_split,
        }
        best_metrics = {
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "best_epoch": int(best_epoch),
            "last_epoch": int(last_epoch),
            "stopped_early": bool(stopped_early),
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

    def _fit_without_validation(self, resume: str | Path | None) -> dict[str, Any]:
        """Train for a fixed epoch count when the only holdout is an untouched test set."""
        model_name = str(self.cfg.model.name).lower()
        metadata = model_metadata(model_name)
        if bool(self.cfg.training.early_stopping.enabled):
            raise ValueError("early stopping requires a validation dataloader")
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise ValueError("ReduceLROnPlateau requires a validation dataloader")

        start_epoch = 1
        if resume is not None:
            checkpoint = load_checkpoint(resume, map_location="cpu")
            restored_epoch, _ = restore_checkpoint(
                checkpoint,
                self.model,
                self.device,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            start_epoch = restored_epoch + 1
            self.logger.info("resumed checkpoint=%s epoch=%d", resume, restored_epoch)

        epochs = int(self.cfg.training.epochs)
        if epochs > 0 and self.optimizer is None:
            raise ValueError("models without trainable parameters require training.epochs=0")
        history: list[dict[str, float | int]] = []
        grad_clip = self.cfg.training.get("grad_clip")
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, epochs + 1):
            last_epoch = epoch
            train_loss = train_one_epoch(
                self.model,
                self.dataloaders["train"],
                self.optimizer,
                self.device,
                grad_clip=float(grad_clip) if grad_clip is not None else None,
            )
            train_metrics = evaluate_isi_nll(self.model, self.dataloaders["train"], self.device)
            train_nll = float(train_metrics["observable_metrics"]["observation_nll"])
            history.append({"epoch": epoch, "train_observation_nll": train_nll})
            self.logger.info(
                "epoch=%d train_loss=%.8f train_observation_nll=%.8f lr=%.6g",
                epoch,
                train_loss,
                train_nll,
                float(self.optimizer.param_groups[0]["lr"]),
            )
            if self.scheduler is not None:
                self.scheduler.step()
            if bool(self.cfg.training.get("save_last", True)):
                save_checkpoint(
                    self.last_path,
                    self.model,
                    epoch,
                    float("nan"),
                    self.cfg,
                    self.optimizer,
                    self.scheduler,
                    trainer_state={"best_epoch": epoch},
                    preprocessing_state=self.preprocessing_state,
                    checkpoint_selection_metric=None,
                )

        save_checkpoint(
            self.best_path,
            self.model,
            last_epoch,
            float("nan"),
            self.cfg,
            self.optimizer,
            self.scheduler,
            trainer_state={"best_epoch": last_epoch},
            preprocessing_state=self.preprocessing_state,
            checkpoint_selection_metric=None,
        )
        if bool(self.cfg.training.get("save_last", True)) and not self.last_path.exists():
            save_checkpoint(
                self.last_path,
                self.model,
                last_epoch,
                float("nan"),
                self.cfg,
                self.optimizer,
                self.scheduler,
                trainer_state={"best_epoch": last_epoch},
                preprocessing_state=self.preprocessing_state,
                checkpoint_selection_metric=None,
            )

        metrics_by_split = {
            split: evaluate_isi_model(self.model, loader, self.device)
            for split, loader in self.dataloaders.items()
        }
        results = {
            "model_name": model_name,
            "result_role": metadata["result_role"],
            "main_result_eligible": bool(metadata["main_result_eligible"]),
            "primary_model": bool(metadata["primary_model"]),
            "fixed_feature_main_baseline": bool(metadata["fixed_feature_main_baseline"]),
            "checkpoint_selection_metric": None,
            "best_epoch": int(last_epoch),
            "last_epoch": int(last_epoch),
            "stopped_early": False,
            "best_checkpoint_path": str(self.best_path),
            "metrics_by_split": metrics_by_split,
        }
        best_metrics = {
            "checkpoint_selection_metric": None,
            "best_epoch": int(last_epoch),
            "last_epoch": int(last_epoch),
            "stopped_early": False,
            "test": metrics_by_split.get("test"),
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
        self.logger.info("final_epoch=%d test_evaluated_once=true", last_epoch)
        return results
