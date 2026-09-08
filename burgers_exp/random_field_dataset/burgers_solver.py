"""Batched Fourier-pseudospectral ETDRK4 solver for viscous Burgers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch


def _etdrk4_coefficients(
    linear: torch.Tensor, dt: float, contour_points: int = 32
) -> tuple[torch.Tensor, ...]:
    real_dtype = linear.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    indices = torch.arange(1, contour_points + 1, device=linear.device, dtype=real_dtype)
    roots = torch.exp(1j * math.pi * (indices - 0.5) / contour_points).to(complex_dtype)
    z = dt * linear.to(complex_dtype)[:, None] + roots[None, :]
    e = torch.exp(dt * linear)
    e2 = torch.exp(0.5 * dt * linear)
    q = dt * ((torch.exp(0.5 * z) - 1.0) / z).mean(1).real
    f1 = dt * ((-4 - z + torch.exp(z) * (4 - 3 * z + z**2)) / z**3).mean(1).real
    f2 = dt * ((2 + z + torch.exp(z) * (-2 + z)) / z**3).mean(1).real
    f3 = dt * ((-4 - 3 * z - z**2 + torch.exp(z) * (4 - z)) / z**3).mean(1).real
    return e, e2, q, f1, f2, f3


def _odd_extension(values: torch.Tensor) -> torch.Tensor:
    values = values.clone()
    values[..., 0] = 0
    values[..., -1] = 0
    return torch.cat((values, -values[..., 1:-1].flip(-1)), dim=-1)


def _automatic_dt(values: torch.Tensor, visc: float, dx: float, T: float) -> float:
    """Choose a conservative advective step; diffusion is handled exactly."""
    max_speed = float(values.abs().max().item())
    advective = 0.2 * dx / max(max_speed, 1.0e-12)
    # The cap also resolves nonlinear interactions when the field is near zero.
    return min(T, advective, 0.01)


@torch.inference_mode()
def solve_burgers(
    initial_condition: np.ndarray | torch.Tensor | Sequence[float],
    visc: float,
    T: float,
    *,
    domain: Sequence[float] = (-1.0, 1.0),
    boundary_type: str = "periodic",
    device: str | torch.device | None = None,
    dt: float | None = None,
) -> np.ndarray | torch.Tensor:
    """Return the Burgers solution at ``T`` for one field or a batch.

    The output type matches the input type. A one-dimensional input returns a
    one-dimensional result; ``(N_sample, Nx)`` is solved in one vectorized batch.
    """
    if visc <= 0 or T < 0:
        raise ValueError("visc must be positive and T must be non-negative")
    if len(domain) != 2 or not domain[0] < domain[1]:
        raise ValueError("domain must be [left, right] with left < right")
    input_is_tensor = isinstance(initial_condition, torch.Tensor)
    requested_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    source = torch.as_tensor(initial_condition)
    dtype = source.dtype if source.is_floating_point() else torch.float32
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    values = source.to(device=requested_device, dtype=dtype)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[None, :]
    if values.ndim != 2 or values.shape[-1] < 3:
        raise ValueError("initial_condition must have shape (Nx,) or (N_sample, Nx)")
    if T == 0:
        result = values[0] if was_vector else values
        return result if input_is_tensor else result.cpu().numpy()

    boundary_type = boundary_type.lower()
    if boundary_type == "dirichlet":
        values = _odd_extension(values)
    elif boundary_type != "periodic":
        raise ValueError("boundary_type must be 'periodic' or 'dirichlet'")

    physical_nx = values.shape[-1]
    length = float(domain[1] - domain[0]) * (2.0 if boundary_type == "dirichlet" else 1.0)
    dx = length / physical_nx
    inferred_dt = _automatic_dt(values, visc, dx, T) if dt is None else float(dt)
    if inferred_dt <= 0:
        raise ValueError("dt must be positive")
    steps = max(1, math.ceil(T / inferred_dt))
    actual_dt = T / steps

    wave_numbers = 2 * math.pi * torch.fft.rfftfreq(
        physical_nx, d=dx, device=requested_device, dtype=values.dtype
    )
    linear = -visc * wave_numbers**2
    e, e2, q, f1, f2, f3 = _etdrk4_coefficients(linear, actual_dt)
    mask = torch.arange(wave_numbers.numel(), device=requested_device) <= physical_nx // 3

    def nonlinear(spectrum: torch.Tensor) -> torch.Tensor:
        field = torch.fft.irfft(spectrum, n=physical_nx, dim=-1)
        return -0.5j * wave_numbers * torch.fft.rfft(field**2, dim=-1) * mask

    spectrum = torch.fft.rfft(values, dim=-1)
    for _ in range(steps):
        nv = nonlinear(spectrum)
        a = e2 * spectrum + q * nv
        na = nonlinear(a)
        b = e2 * spectrum + q * na
        nb = nonlinear(b)
        c = e2 * a + q * (2 * nb - nv)
        nc = nonlinear(c)
        spectrum = e * spectrum + f1 * nv + 2 * f2 * (na + nb) + f3 * nc

    result = torch.fft.irfft(spectrum, n=physical_nx, dim=-1)
    if boundary_type == "dirichlet":
        original_nx = (physical_nx + 2) // 2
        result = result[..., :original_nx]
        result[..., 0] = result[..., -1] = 0
    if was_vector:
        result = result[0]
    return result if input_is_tensor else result.cpu().numpy()


# A short, discoverable alias for callers that prefer ``burgers_solver(...)``.
burgers_solver = solve_burgers
