"""Numerical solvers for supplied ODEs, including process-parallel batches."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

try:
    from .Duffing_oscillators import DuffingEquation
except ImportError:  # Support direct execution from this directory.
    from Duffing_oscillators import DuffingEquation


def solve_equation(
    equation: DuffingEquation,
    evaluation_time: Sequence[float] | np.ndarray,
    *,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Solve one supplied initial-value problem and return displacement values."""
    evaluation_time = np.asarray(evaluation_time, dtype=np.float64)
    if evaluation_time.ndim != 1 or evaluation_time.size < 2:
        raise ValueError("evaluation_time must be a vector with at least two values")
    if evaluation_time[0] < equation.forcing_time[0] or evaluation_time[-1] > equation.forcing_time[-1]:
        raise ValueError("evaluation_time lies outside the supplied forcing interval")
    result = solve_ivp(
        equation.rhs,
        (float(evaluation_time[0]), float(evaluation_time[-1])),
        equation.initial_state,
        method=method,
        t_eval=evaluation_time,
        rtol=rtol,
        atol=atol,
    )
    if not result.success or result.y.shape != (2, evaluation_time.size):
        raise RuntimeError(f"ODE solve failed: {result.message}")
    if not np.all(np.isfinite(result.y)):
        raise FloatingPointError("ODE solve produced non-finite values")
    return result.y[0]


def _solve_job(job: tuple[DuffingEquation, np.ndarray, str, float, float]) -> np.ndarray:
    equation, evaluation_time, method, rtol, atol = job
    return solve_equation(equation, evaluation_time, method=method, rtol=rtol, atol=atol)


def recommended_worker_count() -> int:
    """Use available CPU affinity while keeping per-process SciPy memory bounded."""
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(16, available))


def create_solver_pool(max_workers: int | None = None) -> ProcessPoolExecutor:
    """Create one reusable spawn-based pool for independent ODE trajectories."""
    workers = recommended_worker_count() if max_workers is None else int(max_workers)
    if workers < 1:
        raise ValueError("max_workers must be positive")
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))


def solve_equations_parallel(
    equations: Iterable[DuffingEquation],
    evaluation_time: Sequence[float] | np.ndarray,
    *,
    executor: ProcessPoolExecutor,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    chunksize: int = 1,
) -> np.ndarray:
    """Solve independent equations in parallel using a persistent executor."""
    evaluation_time = np.asarray(evaluation_time, dtype=np.float64)
    jobs = ((equation, evaluation_time, method, rtol, atol) for equation in equations)
    solutions = list(executor.map(_solve_job, jobs, chunksize=chunksize))
    if not solutions:
        return np.empty((0, evaluation_time.size), dtype=np.float64)
    return np.stack(solutions)
