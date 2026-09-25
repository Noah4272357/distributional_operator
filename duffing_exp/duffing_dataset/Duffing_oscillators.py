"""Duffing equation, parameter design, and stochastic-forcing construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.stats import qmc


PARAMETER_NAMES: Final[tuple[str, ...]] = (
    "amplitude",
    "frequency",
    "noise_sigma",
    "correlation_length",
)
PARAMETER_BOUNDS: Final[np.ndarray] = np.asarray(
    ((0.5, 2.0), (0.10, 0.30), (0.05, 0.40), (0.10, 1.00)), dtype=np.float64
)


@dataclass(frozen=True)
class DuffingParameters:
    """Fixed coefficients of y'' + c y' + a y + b y^3 = x(t)."""

    a: float = 1.0
    b: float = 1.0
    c: float = 0.2

    def __post_init__(self) -> None:
        if self.b <= 0 or self.c < 0:
            raise ValueError("b must be positive and c must be non-negative")


@dataclass(frozen=True)
class MeasureParameters:
    """Parameters defining one probability measure over forcing paths."""

    amplitude: float
    frequency: float
    noise_sigma: float
    correlation_length: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (self.amplitude, self.frequency, self.noise_sigma, self.correlation_length),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class DuffingEquation:
    """A picklable Duffing initial-value problem with tabulated forcing."""

    forcing_time: np.ndarray
    forcing_value: np.ndarray
    parameters: DuffingParameters = DuffingParameters()
    initial_state: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        time = np.asarray(self.forcing_time, dtype=np.float64)
        value = np.asarray(self.forcing_value, dtype=np.float64)
        if time.ndim != 1 or value.shape != time.shape or time.size < 2:
            raise ValueError("forcing_time and forcing_value must be equal-length vectors")
        if not np.all(np.diff(time) > 0):
            raise ValueError("forcing_time must be strictly increasing")
        object.__setattr__(self, "forcing_time", time)
        object.__setattr__(self, "forcing_value", value)

    def rhs(self, time: float, state: np.ndarray) -> tuple[float, float]:
        """Evaluate the first-order form of the forced Duffing equation."""
        displacement, velocity = state
        forcing = float(np.interp(time, self.forcing_time, self.forcing_value))
        p = self.parameters
        acceleration = forcing - p.c * velocity - p.a * displacement - p.b * displacement**3
        return float(velocity), float(acceleration)


def latin_hypercube_parameters(count: int, seed: int | None) -> list[MeasureParameters]:
    """Draw a space-filling design over the four prescribed parameter ranges."""
    if count < 1:
        raise ValueError("count must be positive")
    unit_design = qmc.LatinHypercube(d=4, seed=seed).random(count)
    design = qmc.scale(unit_design, PARAMETER_BOUNDS[:, 0], PARAMETER_BOUNDS[:, 1])
    return [MeasureParameters(*row) for row in design]


def make_time_grids(
    final_time: float = 20.0, observation_count: int = 256, solve_dt: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Return the observation grid and an equally spaced fine forcing grid."""
    if final_time <= 0 or observation_count < 2 or solve_dt <= 0:
        raise ValueError("final_time and solve_dt must be positive; observation_count >= 2")
    solve_steps = int(np.ceil(final_time / solve_dt))
    return (
        np.linspace(0.0, final_time, observation_count, dtype=np.float64),
        np.linspace(0.0, final_time, solve_steps + 1, dtype=np.float64),
    )


def _rbf_circulant_spectrum(
    time: np.ndarray, sigma: float, correlation_length: float
) -> tuple[np.ndarray, int]:
    """Build eigenvalues for an exact, adaptively enlarged circulant embedding."""
    if sigma < 0 or correlation_length <= 0:
        raise ValueError("sigma must be non-negative and correlation_length positive")
    dt = float(time[1] - time[0])
    embedding_size = 1 << max(1, (2 * (time.size - 1) - 1).bit_length())
    for _ in range(10):
        half = embedding_size // 2
        lags = dt * np.arange(half + 1)
        covariance = sigma**2 * np.exp(-0.5 * (lags / correlation_length) ** 2)
        first_column = np.concatenate((covariance, covariance[-2:0:-1]))
        eigenvalues = np.fft.rfft(first_column).real
        tolerance = 1e-10 * max(float(eigenvalues.max()), 1.0)
        if float(eigenvalues.min()) >= -tolerance:
            return np.sqrt(np.maximum(eigenvalues, 0.0)), embedding_size
        embedding_size *= 2
    raise RuntimeError("could not construct a positive-semidefinite RBF circulant embedding")


def sample_forcing_paths(
    parameters: MeasureParameters,
    time: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample independent sinusoid-plus-RBF-GP forcing paths on a regular grid."""
    if count < 1:
        raise ValueError("count must be positive")
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size < 2 or not np.allclose(np.diff(time), np.diff(time)[0]):
        raise ValueError("time must be a regular, strictly increasing grid")
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(count, 1))
    periodic = parameters.amplitude * np.sin(
        2.0 * np.pi * parameters.frequency * time[None, :] + phases
    )
    if parameters.noise_sigma == 0:
        return periodic
    spectrum_root, embedding_size = _rbf_circulant_spectrum(
        time, parameters.noise_sigma, parameters.correlation_length
    )
    white_noise = rng.standard_normal((count, embedding_size))
    noise = np.fft.irfft(
        np.fft.rfft(white_noise, axis=1) * spectrum_root[None, :],
        n=embedding_size,
        axis=1,
    )[:, : time.size]
    return periodic + noise
