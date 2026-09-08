"""Central training coordinator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training.pipeline import run_random_field_experiment


class Trainer:
    """Coordinate training, validation, checkpointing, and final evaluation."""

    def __init__(self, cfg: Any, run_dir: str | Path) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)

    def fit(self) -> dict[str, Any]:
        return run_random_field_experiment(self.cfg, output_dir=self.run_dir / "artifacts")
