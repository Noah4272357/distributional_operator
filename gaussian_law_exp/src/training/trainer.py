"""Training coordination, checkpoint selection, and final reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.train_epoch import train_one_epoch
from src.training.validate import evaluate_model
from src.utils.artifacts import (
    BEST_METRICS_ARTIFACT,
    EVAL_BY_SPLIT_ARTIFACT,
    RESULT_ARTIFACT,
    RESULT_TABLE_ARTIFACT,
    TRAINING_HISTORY_ARTIFACT,
    flatten_eval_rows,
    write_history,
    write_json,
    write_rows,
)
from src.utils.checkpoint import (
    CHECKPOINT_SELECTION_METRIC,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from src.utils.metrics import loss_metadata


class Trainer:
    """Coordinate epochs, validation-NLL selection, checkpoints, and artifacts."""

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        dataloaders: dict[str, DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        run_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = run_dir
        self.logger = logger
        self.checkpoint_dir = run_dir / "model_ckpt"
        self.best_path = self.checkpoint_dir / "best.pt"
        self.last_path = self.checkpoint_dir / "last.pt"

    def fit(self, resume: str | Path | None = None) -> dict[str, Any]:
        """Train if needed, restore the best state, evaluate all splits, and report."""
        model_name = str(self.cfg.model.name).lower()
        loss_name = str(self.cfg.loss.name).lower()
        metadata = loss_metadata(loss_name, model_name)
        metrics_cfg = self.cfg.get("metrics")
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

        initial_val = evaluate_model(
            self.model,
            self.dataloaders["val"],
            self.criterion,
            self.device,
            metrics_cfg,
        )
        current_val_nll = float(initial_val["observable_metrics"]["nll"])
        best_val_nll = min(resumed_best, current_val_nll)
        best_epoch = start_epoch - 1
        best_val_metrics = initial_val
        save_checkpoint(
            self.best_path,
            self.model,
            best_epoch,
            best_val_nll,
            self.cfg,
            self.optimizer,
            self.scheduler,
        )

        history: list[dict[str, Any]] = []
        total_epochs = int(self.cfg.training.epochs)
        validation_interval = int(self.cfg.training.get("validation_interval", 10))
        if validation_interval <= 0:
            raise ValueError("training.validation_interval must be positive")
        grad_clip = self.cfg.training.get("grad_clip")
        if self.optimizer is not None:
            for epoch in range(start_epoch, total_epochs + 1):
                train_loss = train_one_epoch(
                    self.model,
                    self.dataloaders["train"],
                    self.optimizer,
                    self.criterion,
                    self.device,
                    grad_clip=float(grad_clip) if grad_clip is not None else None,
                )
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                should_validate = (
                    epoch % validation_interval == 0 or epoch == total_epochs
                )
                if should_validate:
                    val_metrics = evaluate_model(
                        self.model,
                        self.dataloaders["val"],
                        self.criterion,
                        self.device,
                        metrics_cfg,
                    )
                    validation = val_metrics["observable_metrics"]
                    val_nll = float(validation["nll"])
                    history.append(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_training_loss": float(val_metrics["training_loss"]),
                            "val_nll": val_nll,
                            "val_nll_diff": float(validation["nll_diff"]),
                            "val_w2_distance": float(validation["w2_distance"]),
                            "val_kl_divergence": float(validation["kl_divergence"]),
                            "val_hellinger_distance": float(
                                validation["hellinger_distance"]
                            ),
                            "learning_rate": learning_rate,
                        }
                    )
                    self.logger.info(
                        "epoch=%d train_loss=%.8f val_nll=%.8f "
                        "val_nll_diff=%.8f val_w2=%.8f val_kl=%.8f "
                        "val_hellinger=%.8f lr=%.6g",
                        epoch,
                        train_loss,
                        val_nll,
                        validation["nll_diff"],
                        validation["w2_distance"],
                        validation["kl_divergence"],
                        validation["hellinger_distance"],
                        learning_rate,
                    )
                else:
                    self.logger.info(
                        "epoch=%d train_loss=%.8f lr=%.6g",
                        epoch,
                        train_loss,
                        learning_rate,
                    )
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if should_validate:
                        self.scheduler.step(val_nll)
                elif self.scheduler is not None:
                    self.scheduler.step()
                if should_validate and val_nll <= best_val_nll:
                    best_epoch = epoch
                    best_val_nll = val_nll
                    best_val_metrics = val_metrics
                    save_checkpoint(
                        self.best_path,
                        self.model,
                        epoch,
                        best_val_nll,
                        self.cfg,
                        self.optimizer,
                        self.scheduler,
                    )
                if bool(self.cfg.training.get("save_last", True)):
                    save_checkpoint(
                        self.last_path,
                        self.model,
                        epoch,
                        best_val_nll,
                        self.cfg,
                        self.optimizer,
                        self.scheduler,
                    )
        elif bool(self.cfg.training.get("save_last", True)):
            save_checkpoint(
                self.last_path,
                self.model,
                best_epoch,
                best_val_nll,
                self.cfg,
                None,
                None,
            )

        best_checkpoint = load_checkpoint(self.best_path, map_location="cpu")
        restore_checkpoint(best_checkpoint, self.model, self.device)
        metrics_by_split = {
            split: evaluate_model(
                self.model,
                dataloader,
                self.criterion,
                self.device,
                metrics_cfg,
            )
            for split, dataloader in self.dataloaders.items()
        }
        results = {
            "model_name": model_name,
            "loss_name": loss_name,
            "loss_role": metadata["loss_role"],
            "main_result_eligible": bool(metadata["main_result_eligible"]),
            "uses_artificial_pairing": bool(metadata["uses_artificial_pairing"]),
            "pairing_policy": metadata["pairing_policy"],
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "best_epoch": int(best_epoch),
            "best_checkpoint_path": str(self.best_path),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if self.device.type == "cuda"
                else None
            ),
            "metrics_by_split": metrics_by_split,
        }
        best_metrics = {
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "best_epoch": int(best_epoch),
            "val": best_val_metrics,
        }
        rows = flatten_eval_rows(results)
        write_json(self.run_dir / RESULT_ARTIFACT, results)
        write_json(self.run_dir / BEST_METRICS_ARTIFACT, best_metrics)
        write_rows(self.run_dir / RESULT_TABLE_ARTIFACT, rows)
        write_rows(self.run_dir / EVAL_BY_SPLIT_ARTIFACT, rows)
        write_history(self.run_dir / TRAINING_HISTORY_ARTIFACT, history)
        self.logger.info("best_epoch=%d best_val_nll=%.8f", best_epoch, best_val_nll)
        return results
