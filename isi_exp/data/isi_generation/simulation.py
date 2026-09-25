"""LIF simulation and ISI binning used by dataset generation."""

from __future__ import annotations

import torch


def make_isi_bin_edges(*, finite_bins: int, t_max: float) -> torch.Tensor:
    if finite_bins < 1:
        raise ValueError("finite_bins must be positive")
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    return torch.linspace(0.0, float(t_max), int(finite_bins) + 1)


def bin_isi_observations(
    isi_times: torch.Tensor,
    isi_censored: torch.Tensor,
    bin_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite-bin plus tail counts and normalized masses."""
    if isi_times.shape != isi_censored.shape:
        raise ValueError("isi_times and isi_censored must have the same shape")
    finite_bins = bin_edges.numel() - 1
    finite_index = torch.bucketize(isi_times.reshape(-1), bin_edges[1:-1]).clamp_max(finite_bins - 1)
    bin_index = torch.where(
        isi_censored.reshape(-1), torch.full_like(finite_index, finite_bins), finite_index
    )
    counts = torch.zeros(
        (isi_times.shape[0], finite_bins + 1), dtype=torch.float32, device=isi_times.device
    )
    rows = torch.arange(isi_times.shape[0], device=isi_times.device).repeat_interleave(isi_times.shape[1])
    counts.index_put_((rows, bin_index), torch.ones_like(bin_index, dtype=torch.float32), accumulate=True)
    return counts, counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)


def simulate_lif_isi(
    params: torch.Tensor,
    *,
    dt: float,
    t_max: float,
    n_isi: int,
    seed: int,
    v_th: float = 1.0,
    gamma: float = 1.0,
    v0: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Simulate independent reset-trial first-passage times for each law."""
    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("params must have shape [data_size, 2] with columns [m, q]")
    if dt <= 0 or t_max <= 0 or n_isi < 1 or gamma <= 0:
        raise ValueError("dt, t_max, n_isi, and gamma must be positive")
    device, dtype = params.device, params.dtype
    data_size = params.shape[0]
    m = params[:, :1].expand(data_size, n_isi)
    q = params[:, 1:2].expand(data_size, n_isi)
    voltage = torch.full((data_size, n_isi), v0, dtype=dtype, device=device)
    isi_times = torch.full_like(voltage, t_max)
    censored = torch.ones_like(voltage, dtype=torch.bool)
    active = censored.clone()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    max_steps = int(torch.ceil(torch.tensor(t_max / dt)).item())
    for step in range(max_steps):
        t_left = step * dt
        dt_step = min(dt, t_max - t_left)
        if dt_step <= 0:
            break
        noise = torch.randn(voltage.shape, dtype=dtype, device=device, generator=generator)
        next_voltage = voltage + (-voltage / gamma + m) * dt_step + torch.sqrt(q * dt_step) * noise
        crossed = active & (voltage < v_th) & (next_voltage >= v_th)
        if torch.any(crossed):
            fraction = ((v_th - voltage) / (next_voltage - voltage).clamp_min(torch.finfo(dtype).eps)).clamp(0, 1)
            isi_times[crossed] = t_left + dt_step * fraction[crossed]
            censored[crossed] = False
            active &= ~crossed
        if not torch.any(active):
            break
        voltage = torch.where(active, next_voltage, voltage)
    return {"isi_times": isi_times, "isi_censored": censored}
