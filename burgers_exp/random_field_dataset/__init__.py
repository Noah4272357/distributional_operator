"""Random-field initial conditions and a Burgers equation dataset generator."""

from .burgers_solver import solve_burgers
from .generate_initial_condition import (
    StudentTProcess,
    TruncatedGaussianProcess,
    generate_initial_condition,
)

__all__ = [
    "StudentTProcess",
    "TruncatedGaussianProcess",
    "generate_initial_condition",
    "solve_burgers",
]
