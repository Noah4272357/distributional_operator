"""Finite-rank random processes for Burgers initial conditions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class SpectralProcess(ABC):
    """Base class for processes using the requested covariance spectrum."""

    def __init__(
        self,
        trunc_dim: int = 8,
        sigma: float = 1.0,
        alpha: float = 2.0,
        s: float = 2.0,
    ) -> None:
        if trunc_dim < 1:
            raise ValueError("trunc_dim must be positive")
        if sigma <= 0 or alpha <= 0 or s <= 0:
            raise ValueError("sigma, alpha, and s must be positive")
        self.trunc_dim = int(trunc_dim)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.s = float(s)

    @abstractmethod
    def coefficients(
        self, rng: np.random.Generator, shape: tuple[int, ...]
    ) -> NDArray[np.float64]:
        """Draw standardized independent spectral coefficients."""

    def sample(
        self,
        n_sample: int,
        nx: int,
        domain: Sequence[float],
        boundary_type: str,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        if n_sample < 1 or nx < 2:
            raise ValueError("n_sample must be positive and nx must be at least 2")
        if len(domain) != 2 or not domain[0] < domain[1]:
            raise ValueError("domain must be [left, right] with left < right")

        length = float(domain[1] - domain[0])
        modes = np.arange(1, self.trunc_dim + 1, dtype=np.float64)
        # The design specifies lambda_j with j*pi, independently of domain.
        sqrt_eigenvalues = self.sigma * (
            self.alpha**2 + (np.pi * modes) ** 2
        ) ** (-0.5 * self.s)

        if boundary_type == "dirichlet":
            x = np.linspace(domain[0], domain[1], nx, endpoint=True)
            phase = (x - domain[0]) / length
            basis = np.sqrt(2.0) * np.sin(np.pi * modes[:, None] * phase)
            values = (
                self.coefficients(rng, (n_sample, self.trunc_dim))
                * sqrt_eigenvalues
            ) @ basis
            values[:, (0, -1)] = 0.0
            return values

        if boundary_type == "periodic":
            x = np.linspace(domain[0], domain[1], nx, endpoint=False)
            phase = (x - domain[0]) / length
            angles = 2.0 * np.pi * modes[:, None] * phase
            cos_coeff = self.coefficients(rng, (n_sample, self.trunc_dim))
            sin_coeff = self.coefficients(rng, (n_sample, self.trunc_dim))
            # Splitting each eigenvalue equally between sine and cosine keeps
            # the pointwise contribution of a mode at lambda_j.
            scale = sqrt_eigenvalues / np.sqrt(2.0)
            return (cos_coeff * scale) @ np.cos(angles) + (
                sin_coeff * scale
            ) @ np.sin(angles)

        raise ValueError("boundary_type must be 'periodic' or 'dirichlet'")


class TruncatedGaussianProcess(SpectralProcess):
    """Gaussian process truncated to ``trunc_dim`` covariance modes."""

    def coefficients(
        self, rng: np.random.Generator, shape: tuple[int, ...]
    ) -> NDArray[np.float64]:
        return rng.standard_normal(shape)


class StudentTProcess(SpectralProcess):
    """Student-t process using the same covariance kernel as the GP."""

    def __init__(self, nu: float = 4.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if nu <= 2:
            raise ValueError("nu must be greater than 2 for a finite covariance")
        self.nu = float(nu)

    def coefficients(
        self, rng: np.random.Generator, shape: tuple[int, ...]
    ) -> NDArray[np.float64]:
        # Standardize t_nu to unit variance, preserving the Gaussian kernel.
        return rng.standard_t(self.nu, size=shape) * np.sqrt(
            (self.nu - 2.0) / self.nu
        )


_PROCESS_TYPES = {
    "truncated_gaussian": TruncatedGaussianProcess,
    "student_t": StudentTProcess,
}


def generate_initial_condition(
    N_sample: int,
    Nx: int = 512,
    domain: Sequence[float] = (-1.0, 1.0),
    boundary_type: str = "periodic",
    process_type: str = "truncated_Gaussian",
    seed: int | None = None,
    **process_args: Any,
) -> NDArray[np.float64]:
    """Return ``N_sample`` random initial conditions with shape ``(N_sample, Nx)``."""
    key = process_type.lower()
    if key not in _PROCESS_TYPES:
        supported = ", ".join(sorted(_PROCESS_TYPES))
        raise ValueError(f"unknown process_type {process_type!r}; choose from {supported}")
    process = _PROCESS_TYPES[key](**process_args)
    return process.sample(
        N_sample, Nx, domain, boundary_type.lower(), np.random.default_rng(seed)
    )
