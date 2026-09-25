"""Dataset view over a compact generated ISI payload."""

from __future__ import annotations

from typing import Any

import torch


class ISILawDataset(torch.utils.data.Dataset):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __len__(self) -> int:
        return int(self.payload["law_ids"].numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "law_id": self.payload["law_ids"][index],
            "regime_label": self.payload["regime_labels"][index],
            "params": self.payload["params"][index],
            "normalized_params": self.payload["normalized_params"][index],
            "input_particles": self.payload["input_particles"][index],
            "input_features": self.payload["input_features"][index],
            "bin_counts": self.payload["bin_counts"][index],
            "empirical_bin_mass": self.payload["empirical_bin_mass"][index],
            "bin_edges": self.payload["bin_edges"],
        }
        if "process_features" in self.payload:
            item["process_features"] = self.payload["process_features"][index]
        return item
