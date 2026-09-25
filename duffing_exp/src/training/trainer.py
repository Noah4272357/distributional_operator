"""Central training coordinator."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.checkpoint import save_checkpoint
from src.utils.logging import MetricsLogger

from .train_epoch import train_one_epoch
from .validate import validate


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        device: torch.device,
        config: dict,
        run_dir: Path,
        start_epoch: int = 0,
        best_validation: float = float("inf"),
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.config = config
        self.run_dir = run_dir
        self.start_epoch = start_epoch
        self.best_validation = best_validation
        self.logger = MetricsLogger(run_dir)

    def fit(self) -> dict[str, float]:
        training_cfg = self.config["training"]
        epochs = int(training_cfg["epochs"])
        grad_clip = training_cfg.get("grad_clip")
        log_every = int(training_cfg["log_every"])
        early_stopping_cfg = training_cfg.get("early_stopping") or {}
        early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
        early_stopping_patience = int(early_stopping_cfg.get("patience", 10))
        early_stopping_min_delta = float(
            early_stopping_cfg.get("min_delta", 0.0)
        )
        early_stopping_best = self.best_validation
        epochs_without_improvement = 0
        checkpoint_cfg = self.config["checkpoint"]
        start = time.perf_counter()
        last_metrics: dict[str, float] = {}
        epochs_run = 0
        stopped_early = False
        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.perf_counter()
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                None if grad_clip is None else float(grad_clip),
            )
            validation_metrics = validate(
                self.model, self.validation_loader, self.criterion, self.device
            )
            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(validation_metrics["sinkhorn_distance"])
            else:
                self.scheduler.step()
            last_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_sinkhorn_distance": validation_metrics[
                    "sinkhorn_distance"
                ],
                "validation_mmd": validation_metrics["mmd"],
                "validation_sliced_wasserstein": validation_metrics[
                    "sliced_wasserstein"
                ],
                "validation_energy_distance": validation_metrics["energy_distance"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            self.logger.log(last_metrics)
            epochs_run += 1
            validation_sinkhorn = float(validation_metrics["sinkhorn_distance"])
            improved = validation_sinkhorn < self.best_validation
            if improved:
                self.best_validation = validation_sinkhorn
                if checkpoint_cfg["save_best"]:
                    save_checkpoint(
                        self.run_dir / "best.pt", epoch, self.model, self.optimizer,
                        self.scheduler, self.best_validation, self.config
                    )
            if checkpoint_cfg["save_last"]:
                save_checkpoint(
                    self.run_dir / "last.pt", epoch, self.model, self.optimizer,
                    self.scheduler, self.best_validation, self.config
                )
            if (epoch + 1) % log_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} train={train_loss:.8e} "
                    f"sinkhorn={validation_metrics['sinkhorn_distance']:.8e} "
                    f"mmd={validation_metrics['mmd']:.8e} "
                    f"sliced_wasserstein={validation_metrics['sliced_wasserstein']:.8e} "
                    f"energy_distance={validation_metrics['energy_distance']:.8e} "
                    f"lr={last_metrics['lr']:.8e} seconds={last_metrics['epoch_seconds']:.3f}",
                    flush=True,
                )
            if early_stopping_enabled:
                if validation_sinkhorn < (
                    early_stopping_best - early_stopping_min_delta
                ):
                    early_stopping_best = validation_sinkhorn
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    stopped_early = True
                    print(
                        f"early stopping at epoch {epoch + 1}: "
                        f"validation sinkhorn did not improve by more than "
                        f"{early_stopping_min_delta:.8e} for "
                        f"{early_stopping_patience} consecutive epochs",
                        flush=True,
                    )
                    break
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary = {
            **last_metrics,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "training_seconds": elapsed,
            "seconds_per_epoch": elapsed / max(epochs_run, 1),
            "estimated_1000_epoch_hours": elapsed / max(epochs_run, 1) * 1000 / 3600,
            "peak_process_rss_bytes": int(peak_rss_kib * 1024),
            "final_process_rss_bytes": psutil.Process().memory_info().rss,
            "peak_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
            ),
            "peak_gpu_reserved_bytes": (
                torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0
            ),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
