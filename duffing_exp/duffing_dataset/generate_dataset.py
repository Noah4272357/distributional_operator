"""Generate distribution-to-distribution Duffing data in streaming HDF5 format."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np

try:
    from .Duffing_oscillators import (
        PARAMETER_NAMES,
        DuffingEquation,
        DuffingParameters,
        latin_hypercube_parameters,
        make_time_grids,
        sample_forcing_paths,
    )
    from .ode_solver import create_solver_pool, recommended_worker_count, solve_equations_parallel
except ImportError:  # Support ``python generate_dataset.py`` on a server.
    from Duffing_oscillators import (
        PARAMETER_NAMES,
        DuffingEquation,
        DuffingParameters,
        latin_hypercube_parameters,
        make_time_grids,
        sample_forcing_paths,
    )
    from ode_solver import create_solver_pool, recommended_worker_count, solve_equations_parallel


def _linux_process_tree_rss(root_pid: int) -> int:
    """Return aggregate resident bytes for a Linux process and all descendants."""
    pending, seen, total_kib = [root_pid], set(), 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                total_kib += int(line.split()[1])
                break
        pending.extend(map(int, children))
    return total_kib * 1024


class PeakMemoryMonitor:
    """Low-overhead aggregate process-tree RSS sampler for remote benchmarks."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            if sys.platform.startswith("linux"):
                self.peak_bytes = max(self.peak_bytes, _linux_process_tree_rss(os.getpid()))
            else:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                self.peak_bytes = max(self.peak_bytes, int(rss) * 1024)

    def __enter__(self) -> "PeakMemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        if sys.platform.startswith("linux"):
            self.peak_bytes = max(self.peak_bytes, _linux_process_tree_rss(os.getpid()))


def generate_dataset(
    output_path: str | Path,
    *,
    num_measures: int = 1400,
    n_input: int = 128,
    n_output: int = 128,
    observation_count: int = 256,
    final_time: float = 20.0,
    solve_dt: float = 0.01,
    seed: int = 2025,
    workers: int | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    overwrite: bool = False,
) -> Path:
    """Generate the prescribed dataset and return its final HDF5 path."""
    if min(num_measures, n_input, n_output) < 1:
        raise ValueError("num_measures, n_input, and n_output must be positive")
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".inprogress")
    if temporary_path.exists():
        raise FileExistsError(f"incomplete output already exists: {temporary_path}")

    worker_count = recommended_worker_count() if workers is None else int(workers)
    observation_time, forcing_time = make_time_grids(final_time, observation_count, solve_dt)
    parameters = latin_hypercube_parameters(num_measures, seed)
    master_rng = np.random.default_rng(seed)
    input_seeds = master_rng.integers(0, 2**63 - 1, size=num_measures, dtype=np.int64)
    output_seeds = master_rng.integers(0, 2**63 - 1, size=num_measures, dtype=np.int64)
    fixed = DuffingParameters()
    shape_x = (num_measures, n_input, observation_count)
    shape_y = (num_measures, n_output, observation_count)
    started = time.perf_counter()

    print(
        f"Generating {num_measures} measures ({n_input} input + {n_output} output paths each) "
        f"with {worker_count} ODE workers",
        flush=True,
    )
    try:
        with PeakMemoryMonitor() as memory, create_solver_pool(worker_count) as executor, h5py.File(
            temporary_path, "w", libver="latest"
        ) as handle:
            x_data = handle.create_dataset(
                "X", shape=shape_x, dtype="f4", chunks=(1, n_input, observation_count), compression="lzf"
            )
            y_data = handle.create_dataset(
                "Y", shape=shape_y, dtype="f4", chunks=(1, n_output, observation_count), compression="lzf"
            )
            handle.create_dataset(
                "theta", data=np.stack([item.as_array() for item in parameters]).astype(np.float32)
            )
            handle.create_dataset("time", data=observation_time)
            handle.create_dataset("input_seed", data=input_seeds)
            handle.create_dataset("output_seed", data=output_seeds)
            handle.attrs.update(
                axis_order="measure,sample,time",
                parameter_names=json.dumps(PARAMETER_NAMES),
                duffing_parameters=json.dumps(vars(fixed)),
                forcing="A*sin(2*pi*f*t+phase) + GP(0, RBF(sigma, correlation_length))",
                independent_input_output_ensembles=True,
                seed=seed,
                solve_dt=solve_dt,
                solver="scipy.integrate.solve_ivp:RK45",
                rtol=rtol,
                atol=atol,
                workers=worker_count,
            )

            for index, measure in enumerate(parameters):
                input_paths_fine = sample_forcing_paths(
                    measure, forcing_time, n_input, np.random.default_rng(int(input_seeds[index]))
                )
                x_data[index] = np.stack(
                    [np.interp(observation_time, forcing_time, path) for path in input_paths_fine]
                ).astype(np.float32)
                del input_paths_fine

                output_forcing = sample_forcing_paths(
                    measure, forcing_time, n_output, np.random.default_rng(int(output_seeds[index]))
                )
                equations = (
                    DuffingEquation(forcing_time, output_forcing[row], fixed)
                    for row in range(n_output)
                )
                y_data[index] = solve_equations_parallel(
                    equations,
                    observation_time,
                    executor=executor,
                    rtol=rtol,
                    atol=atol,
                    chunksize=max(1, n_output // (worker_count * 4)),
                ).astype(np.float32)
                del output_forcing
                elapsed = time.perf_counter() - started
                print(
                    f"measure {index + 1}/{num_measures}; elapsed={elapsed:.1f}s; "
                    f"ETA={elapsed / (index + 1) * (num_measures - index - 1):.1f}s",
                    flush=True,
                )

            handle.attrs["completed_measures"] = num_measures
            handle.flush()
        elapsed = time.perf_counter() - started
        with h5py.File(temporary_path, "r+") as handle:
            handle.attrs["generation_seconds"] = elapsed
            handle.attrs["peak_process_tree_rss_bytes"] = memory.peak_bytes
        os.replace(temporary_path, output_path)
    except BaseException:
        print(f"Generation interrupted; partial file retained at {temporary_path}", file=sys.stderr)
        raise

    size_bytes = output_path.stat().st_size
    print(f"Saved {output_path}", flush=True)
    print(f"generation_seconds={elapsed:.6f}", flush=True)
    print(f"peak_process_tree_rss_bytes={memory.peak_bytes}", flush=True)
    print(f"hdf5_bytes={size_bytes}", flush=True)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("duffing_dataset.h5"))
    parser.add_argument("--num-measures", type=int, default=1400)
    parser.add_argument("--n-input", type=int, default=128)
    parser.add_argument("--n-output", type=int, default=128)
    parser.add_argument("--observation-count", type=int, default=256)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--solve-dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    generate_dataset(
        arguments.output,
        num_measures=arguments.num_measures,
        n_input=arguments.n_input,
        n_output=arguments.n_output,
        observation_count=arguments.observation_count,
        final_time=arguments.final_time,
        solve_dt=arguments.solve_dt,
        seed=arguments.seed,
        workers=arguments.workers,
        rtol=arguments.rtol,
        atol=arguments.atol,
        overwrite=arguments.overwrite,
    )
