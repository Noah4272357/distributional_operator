"""Loss construction for categorical ISI observations."""

import torch
import torch.nn.functional as F


def compute_isi_loss(prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute count-weighted per-observed-event multinomial NLL."""
    counts = batch["bin_counts"].to(dtype=prediction["logits"].dtype)
    log_probs = F.log_softmax(prediction["logits"], dim=-1)
    per_law_events = counts.sum(dim=-1).clamp_min(1.0)
    return (-(counts * log_probs).sum(dim=-1) / per_law_events).mean()

build_loss = lambda: compute_isi_loss
