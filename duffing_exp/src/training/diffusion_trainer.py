"""Training coordinator dedicated to conditional diffusion models."""

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

from .diffusion_train_epoch import train_diffusion_one_epoch
from .diffusion_validate import validate_diffusion


class DiffusionTrainer:
    """Coordinate DDPM training while preserving common experiment artifacts."""

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
        if not callable(getattr(model, "diffusion_loss", None)):
            raise TypeError(
                "DiffusionTrainer requires a model with "
                "diffusion_loss(inputs, targets)"
            )
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.distribution_criterion = criterion
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.config = config
        self.run_dir = run_dir
        self.start_epoch = start_epoch
        self.best_validation = best_validation
        self.logger = MetricsLogger(run_dir)

    def fit(self) -> dict[str, float]:
        epochs = int(self.config["training"]["epochs"])
        grad_clip = self.config["training"].get("grad_clip")
        log_every = int(self.config["training"]["log_every"])
        checkpoint_config = self.config["checkpoint"]
        start = time.perf_counter()
        last_metrics: dict[str, float] = {}
        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.perf_counter()
            train_loss = train_diffusion_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.device,
                None if grad_clip is None else float(grad_clip),
            )
            validation_metrics = validate_diffusion(
                self.model,
                self.validation_loader,
                self.distribution_criterion,
                self.device,
            )
            validation_loss = validation_metrics["diffusion_loss"]
            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(validation_loss)
            else:
                self.scheduler.step()
            last_metrics = {
                "epoch": epoch + 1,
                "train_diffusion_loss": train_loss,
                "validation_diffusion_loss": validation_loss,
                "validation_sinkhorn_distance": validation_metrics[
                    "sinkhorn_distance"
                ],
                "validation_mmd": validation_metrics["mmd"],
                "validation_sliced_wasserstein": validation_metrics[
                    "sliced_wasserstein"
                ],
                "validation_energy_distance": validation_metrics[
                    "energy_distance"
                ],
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            self.logger.log(last_metrics)
            improved = validation_loss < self.best_validation
            if improved:
                self.best_validation = validation_loss
                if checkpoint_config["save_best"]:
                    save_checkpoint(
                        self.run_dir / "best.pt",
                        epoch,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.best_validation,
                        self.config,
                    )
            if checkpoint_config["save_last"]:
                save_checkpoint(
                    self.run_dir / "last.pt",
                    epoch,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.best_validation,
                    self.config,
                )
            if (epoch + 1) % log_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} "
                    f"train_diffusion={train_loss:.8e} "
                    f"validation_diffusion={validation_loss:.8e} "
                    f"sinkhorn={validation_metrics['sinkhorn_distance']:.8e} "
                    f"mmd={validation_metrics['mmd']:.8e} "
                    f"sliced_wasserstein="
                    f"{validation_metrics['sliced_wasserstein']:.8e} "
                    f"energy_distance="
                    f"{validation_metrics['energy_distance']:.8e} "
                    f"lr={last_metrics['lr']:.8e} "
                    f"seconds={last_metrics['epoch_seconds']:.3f}",
                    flush=True,
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        epochs_run = epochs - self.start_epoch
        summary = {
            **last_metrics,
            "epochs_run": epochs_run,
            "training_seconds": elapsed,
            "seconds_per_epoch": elapsed / max(epochs_run, 1),
            "estimated_1000_epoch_hours": elapsed
            / max(epochs_run, 1)
            * 1000
            / 3600,
            "peak_process_rss_bytes": int(peak_rss_kib * 1024),
            "final_process_rss_bytes": psutil.Process().memory_info().rss,
            "peak_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            "peak_gpu_reserved_bytes": (
                torch.cuda.max_memory_reserved(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
