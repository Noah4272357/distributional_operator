"""CSV metrics logging."""

from __future__ import annotations

import csv
from pathlib import Path


class MetricsLogger:
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "metrics.csv"

    def log(self, metrics: dict[str, float]) -> None:
        first = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics))
            if first:
                writer.writeheader()
            writer.writerow(metrics)
