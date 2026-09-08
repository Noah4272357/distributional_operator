"""Generate random-field/Burgers pairs from a JSON configuration."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

try:
    from .burgers_solver import solve_burgers
    from .generate_initial_condition import generate_initial_condition
except ImportError:  # Support ``python generate_dataset.py ...`` on a server.
    from burgers_solver import solve_burgers
    from generate_initial_condition import generate_initial_condition


def _sample_range(spec: Any, rng: np.random.Generator) -> Any:
    if not isinstance(spec, dict) or "distribution" not in spec:
        return spec
    distribution = spec["distribution"].lower()
    low, high = spec["range"]
    if distribution == "uniform":
        return float(rng.uniform(low, high))
    if distribution == "loguniform":
        if low <= 0 or high <= 0:
            raise ValueError("loguniform bounds must be positive")
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    raise ValueError(f"unsupported parameter distribution {distribution!r}")


def generate_dataset(config_path: str | Path) -> Path:
    """Generate the configured dataset and return its HDF5 path."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    rng = np.random.default_rng(config.get("seed"))
    initial_config = config["initial_condition"]
    process_type = config.get("process_type", "truncated_Gaussian")
    ranges = config["process_args_range"][process_type]
    n_process = int(config["N_process"])
    n_sample = int(initial_config["N_sample"])
    if n_process < 1 or n_sample < 1:
        raise ValueError("N_process and N_sample must be positive")

    initial_batches: list[np.ndarray] = []
    sampled_parameters: list[dict[str, Any]] = []
    for process_index in range(n_process):
        args = {name: _sample_range(spec, rng) for name, spec in ranges.items()}
        sampled_parameters.append(args)
        initial_batches.append(
            generate_initial_condition(
                n_sample,
                Nx=int(initial_config.get("Nx", 512)),
                domain=initial_config.get("domain", [-1, 1]),
                boundary_type=initial_config.get("boundary_type", "periodic"),
                process_type=process_type,
                seed=int(rng.integers(0, 2**32 - 1)),
                **args,
            )
        )
    initial = np.concatenate(initial_batches, axis=0)

    solver = config["solver"]
    solution = solve_burgers(
        initial,
        float(solver["visc"]),
        float(solver["T"]),
        domain=initial_config.get("domain", [-1, 1]),
        boundary_type=initial_config.get("boundary_type", "periodic"),
        device=solver.get("device"),
        dt=solver.get("dt"),
    )

    output_path = Path(config.get("output_path", "random_field_dataset/dataset.h5"))
    if not output_path.is_absolute():
        output_path = (config_path.parent / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("initial", data=initial.astype(np.float32), compression="gzip")
        handle.create_dataset("solution", data=np.asarray(solution, dtype=np.float32), compression="gzip")
        handle.attrs["config"] = json.dumps(config)
        handle.attrs["process_parameters"] = json.dumps(sampled_parameters)
        handle.attrs["compute_device"] = str(
            solver.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
    return output_path


def _parameter_grid(config: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    """Sample values per range and return their Cartesian product."""
    count_config = config["parameter_values_per_arg"]
    sampled: dict[str, list[Any]] = {}
    for name, spec in config["process_args_range"].items():
        if not isinstance(spec, dict) or "distribution" not in spec:
            sampled[name] = [spec]
        else:
            count = int(count_config[name] if isinstance(count_config, dict) else count_config)
            if count < 1:
                raise ValueError(f"parameter sample count for {name} must be positive")
            sampled[name] = [_sample_range(spec, rng) for _ in range(count)]
    names = list(sampled)
    return [dict(zip(names, values)) for values in itertools.product(*(sampled[n] for n in names))]


def _copy_rows(source: h5py.Dataset, target: h5py.Dataset, batch_size: int = 10) -> None:
    """Copy a process-major HDF5 dataset without loading it all into memory."""
    for start in range(0, source.shape[0], batch_size):
        stop = min(start + batch_size, source.shape[0])
        target[start:stop] = source[start:stop]


def _json_history(handle: h5py.File, plural: str, singular: str) -> list[Any]:
    """Read new-style metadata history or upgrade a legacy single value."""
    if plural in handle.attrs:
        return list(json.loads(handle.attrs[plural]))
    if singular in handle.attrs:
        return [json.loads(handle.attrs[singular])]
    return []


def _sample_split_experiment(
    config_path: Path,
    config: dict[str, Any],
    rng: np.random.Generator,
    parameters: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Generate all processes, splitting each process along its sample axis."""
    process_type = config.get("process_type", "student_t")
    if process_type.lower() != "student_t":
        raise ValueError("data_generation.md requires process_type='student_t'")
    n_sample = int(config["N_sample"])
    stored_samples = int(config["stored_samples_per_process"])
    if n_sample != 2 * stored_samples:
        raise ValueError("N_sample must equal twice stored_samples_per_process")
    nx = int(config["Nx"])
    domain = config.get("domain", [-1.0, 1.0])
    boundary = config.get("boundary_type", "periodic")
    batch_processes = int(config.get("solver_batch_processes", 2))
    nu = float(config.get("student_t_nu", 4.0))
    output_dir = Path(config.get("output_dir", "."))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_path = output_dir / "initial.hdf5"
    solution_path = output_dir / "solution.hdf5"
    initial_tmp = initial_path.with_suffix(initial_path.suffix + ".inprogress")
    solution_tmp = solution_path.with_suffix(solution_path.suffix + ".inprogress")
    shape = (len(parameters), stored_samples, nx)

    if initial_path.exists() or solution_path.exists():
        if not initial_path.exists() or not solution_path.exists():
            raise FileExistsError("only one final HDF5 file exists; refusing an ambiguous overwrite")
        with h5py.File(initial_path, "r") as initial_handle, h5py.File(
            solution_path, "r"
        ) as solution_handle:
            complete = (
                "initial" in initial_handle
                and "solution" in solution_handle
                and initial_handle["initial"].shape == shape
                and solution_handle["solution"].shape == shape
                and json.loads(initial_handle.attrs["config"]) == config
                and json.loads(solution_handle.attrs["config"]) == config
            )
        if complete:
            print(f"Sample-split dataset already complete at {output_dir}", flush=True)
            return initial_path, solution_path
        raise FileExistsError("final HDF5 files already exist with a different configuration")

    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()

    common_attrs = {
        "config": json.dumps(config),
        "parameter_grid": json.dumps(parameters),
        "axis_order": "process,sample,space",
        "process_type": process_type,
        "sample_partition": json.dumps(
            {"initial": [0, stored_samples], "solution_source": [stored_samples, n_sample]}
        ),
    }
    started = time.perf_counter()
    print(
        f"Generating {len(parameters)} Student-t processes with {n_sample} samples each; "
        f"saving {stored_samples} initial and solving {stored_samples} samples per process",
        flush=True,
    )
    with h5py.File(initial_tmp, "w", libver="latest") as initial_handle, h5py.File(
        solution_tmp, "w", libver="latest"
    ) as solution_handle:
        initial_data = initial_handle.create_dataset(
            "initial", shape=shape, dtype="f4",
            chunks=(1, stored_samples, nx), compression="gzip",
        )
        solution_data = solution_handle.create_dataset(
            "solution", shape=shape, dtype="f4",
            chunks=(1, stored_samples, nx), compression="gzip",
        )
        initial_indices = initial_handle.create_dataset(
            "parameter_index", data=np.arange(len(parameters), dtype=np.int32)
        )
        solution_indices = solution_handle.create_dataset(
            "parameter_index", data=np.arange(len(parameters), dtype=np.int32)
        )
        del initial_indices, solution_indices
        sample_seeds = rng.integers(0, 2**32 - 1, size=len(parameters), dtype=np.uint32)
        initial_handle.create_dataset("sample_seed", data=sample_seeds)
        solution_handle.create_dataset("sample_seed", data=sample_seeds)

        for start in range(0, len(parameters), batch_processes):
            stop = min(start + batch_processes, len(parameters))
            initial_batch: list[np.ndarray] = []
            solver_batch: list[np.ndarray] = []
            for index in range(start, stop):
                fields = generate_initial_condition(
                    n_sample, Nx=nx, domain=domain, boundary_type=boundary,
                    process_type=process_type, seed=int(sample_seeds[index]),
                    **parameters[index], nu=nu,
                ).astype(np.float32)
                initial_batch.append(fields[:stored_samples])
                solver_batch.append(fields[stored_samples:])
            initial_array = np.stack(initial_batch)
            solver_array = np.stack(solver_batch)
            initial_data[start:stop] = initial_array
            flat = torch.from_numpy(solver_array.reshape(-1, nx))
            solved = solve_burgers(
                flat, float(config["solver"]["visc"]), float(config["solver"]["T"]),
                domain=domain, boundary_type=boundary, device=device,
                dt=config["solver"].get("dt"),
            )
            solution_data[start:stop] = (
                solved.detach().cpu().numpy().reshape(solver_array.shape)
            )
            if stop % 50 == 0 or stop == len(parameters):
                print(f"process generation and solve: {stop}/{len(parameters)}", flush=True)

        elapsed = time.perf_counter() - started
        for handle in (initial_handle, solution_handle):
            for name, value in common_attrs.items():
                handle.attrs[name] = value
            handle.attrs["generation_seconds"] = elapsed
            handle.attrs["device"] = str(device)
            if str(device).startswith("cuda") and torch.cuda.is_available():
                handle.attrs["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated())

    os.replace(initial_tmp, initial_path)
    os.replace(solution_tmp, solution_path)
    print(f"Sample-split generation seconds: {elapsed:.3f}", flush=True)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        print(f"Peak allocated GPU bytes: {torch.cuda.max_memory_allocated()}", flush=True)
    return initial_path, solution_path


def generate_large_experiment(config_path: str | Path) -> tuple[Path, Path]:
    """Generate the Student-t datasets described in ``data_generation.md``.

    The 1,000 Cartesian-product parameter sets are randomly ordered and split
    into disjoint halves. The first half is stored as initial conditions. The
    second half is evolved with Burgers and only its final-time solution is
    stored. Temporary files are atomically promoted after successful writes.
    """
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    rng = np.random.default_rng(config.get("seed", 42))
    parameters = _parameter_grid(config, rng)
    count_config = config["parameter_values_per_arg"]
    if isinstance(count_config, dict):
        expected = int(np.prod(list(count_config.values())))
    else:
        expected = int(count_config) ** 3
    if len(parameters) != expected:
        raise ValueError(f"expected {expected} parameter combinations, got {len(parameters)}")

    if config.get("split_samples", False):
        return _sample_split_experiment(config_path, config, rng, parameters)

    process_type = config.get("process_type", "student_t")
    if process_type.lower() != "student_t":
        raise ValueError("data_generation.md requires process_type='student_t'")
    direct_count = int(config["direct_processes"])
    solved_count = int(config["solved_processes"])
    if direct_count + solved_count > len(parameters):
        raise ValueError("requested more processes than the parameter grid contains")
    if direct_count + solved_count != len(parameters):
        raise ValueError("direct_processes + solved_processes must use all parameter sets")
    n_sample = int(config["N_sample"])
    nx = int(config["Nx"])
    batch_processes = int(config.get("solver_batch_processes", 2))
    domain = config.get("domain", [-1.0, 1.0])
    boundary = config.get("boundary_type", "periodic")
    output_dir = Path(config.get("output_dir", "."))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_path = output_dir / "initial.hdf5"
    solution_path = output_dir / "solution.hdf5"
    initial_tmp = initial_path.with_suffix(initial_path.suffix + ".inprogress")
    solution_tmp = solution_path.with_suffix(solution_path.suffix + ".inprogress")

    merge_existing = bool(config.get("merge_existing", False))
    old_count = 0
    generation_id = 0
    config_history: list[Any] = []
    grid_history: list[Any] = []
    if merge_existing:
        if not initial_path.exists() or not solution_path.exists():
            raise FileNotFoundError("merge_existing requires both existing HDF5 files")
        with h5py.File(initial_path, "r") as old_initial, h5py.File(
            solution_path, "r"
        ) as old_solution:
            if set(old_initial) < {"initial", "parameter_index"}:
                raise ValueError("existing initial.hdf5 has an incompatible layout")
            if set(old_solution) < {"solution", "parameter_index"}:
                raise ValueError("existing solution.hdf5 has an incompatible layout")
            if old_initial["initial"].shape != old_solution["solution"].shape:
                raise ValueError("existing initial and solution shapes do not match")
            old_count, old_samples, old_nx = old_initial["initial"].shape
            if (old_samples, old_nx) != (n_sample, nx):
                raise ValueError("existing files do not match configured N_sample/Nx")
            target_count = int(config["target_processes_per_file"])
            if old_count == target_count:
                print(f"Merged target already complete: {target_count} processes per file", flush=True)
                return initial_path, solution_path
            if old_count + direct_count != target_count or old_count + solved_count != target_count:
                raise ValueError(
                    "existing rows plus newly generated rows must equal target_processes_per_file"
                )
            old_ids = old_initial.get("generation_id")
            generation_id = int(np.max(old_ids[:])) + 1 if old_ids is not None else 1
            config_history = _json_history(old_initial, "configs", "config")
            grid_history = _json_history(old_initial, "parameter_grids", "parameter_grid")

    order = rng.permutation(len(parameters))
    nu = float(config.get("student_t_nu", 4.0))
    direct_specs = [({**parameters[index], "nu": nu}, int(index)) for index in order[:direct_count]]
    solved_specs = [
        ({**parameters[index], "nu": nu}, int(index))
        for index in order[direct_count : direct_count + solved_count]
    ]

    shape_direct = (old_count + direct_count, n_sample, nx)
    shape_solved = (old_count + solved_count, n_sample, nx)
    config_history.append(config)
    grid_history.append(parameters)
    common_attrs = {
        "config": json.dumps(config),
        "parameter_grid": json.dumps(parameters),
        "configs": json.dumps(config_history),
        "parameter_grids": json.dumps(grid_history),
        "axis_order": "process,sample,space",
        "process_type": process_type,
        "generation_count": generation_id + 1,
    }
    started = time.perf_counter()
    print(
        f"Writing {old_count} old + {direct_count} new Student-t initial processes "
        f"into {initial_tmp}",
        flush=True,
    )
    with h5py.File(initial_tmp, "w", libver="latest") as handle:
        dataset = handle.create_dataset(
            "initial", shape=shape_direct, dtype="f4",
            chunks=(1, n_sample, nx), compression="gzip",
        )
        parameter_index = handle.create_dataset(
            "parameter_index", shape=(shape_direct[0],), dtype="i4"
        )
        generation_ids = handle.create_dataset(
            "generation_id", shape=(shape_direct[0],), dtype="i4"
        )
        if merge_existing:
            with h5py.File(initial_path, "r") as old:
                _copy_rows(old["initial"], dataset)
                parameter_index[:old_count] = old["parameter_index"][:]
                if "generation_id" in old:
                    generation_ids[:old_count] = old["generation_id"][:]
                else:
                    generation_ids[:old_count] = 0
        for dest, (args, source_index) in enumerate(direct_specs, old_count):
            dataset[dest] = generate_initial_condition(
                n_sample, Nx=nx, domain=domain, boundary_type=boundary,
                process_type=process_type,
                seed=int(rng.integers(0, 2**32 - 1)), **args,
            ).astype(np.float32)
            parameter_index[dest] = source_index
            generation_ids[dest] = generation_id
            generated = dest - old_count + 1
            if generated % 50 == 0:
                print(f"new initial generation: {generated}/{direct_count}", flush=True)
        direct_seconds = time.perf_counter() - started
        for name, value in common_attrs.items():
            handle.attrs[name] = value
        handle.attrs["generation_seconds"] = direct_seconds
    if not merge_existing:
        os.replace(initial_tmp, initial_path)

    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    solver_started = time.perf_counter()
    print(
        f"Writing {old_count} old + {solved_count} new Student-t solutions on {device} "
        f"into {solution_tmp}",
        flush=True,
    )
    with h5py.File(solution_tmp, "w", libver="latest") as handle:
        solution_data = handle.create_dataset(
            "solution", shape=shape_solved, dtype="f4",
            chunks=(1, n_sample, nx), compression="gzip",
        )
        parameter_index = handle.create_dataset(
            "parameter_index", shape=(shape_solved[0],), dtype="i4"
        )
        generation_ids = handle.create_dataset(
            "generation_id", shape=(shape_solved[0],), dtype="i4"
        )
        if merge_existing:
            with h5py.File(solution_path, "r") as old:
                _copy_rows(old["solution"], solution_data)
                parameter_index[:old_count] = old["parameter_index"][:]
                if "generation_id" in old:
                    generation_ids[:old_count] = old["generation_id"][:]
                else:
                    generation_ids[:old_count] = 0
        for relative_start in range(0, solved_count, batch_processes):
            relative_stop = min(relative_start + batch_processes, solved_count)
            start = old_count + relative_start
            stop = old_count + relative_stop
            batch = []
            for dest, (args, source_index) in enumerate(
                solved_specs[relative_start:relative_stop], start
            ):
                batch.append(generate_initial_condition(
                    n_sample, Nx=nx, domain=domain, boundary_type=boundary,
                    process_type=process_type,
                    seed=int(rng.integers(0, 2**32 - 1)), **args,
                ).astype(np.float32))
                parameter_index[dest] = source_index
                generation_ids[dest] = generation_id
            batch_array = np.stack(batch)
            flat = torch.from_numpy(batch_array.reshape(-1, nx))
            solved = solve_burgers(
                flat, float(config["solver"]["visc"]), float(config["solver"]["T"]),
                domain=domain, boundary_type=boundary, device=device, dt=config["solver"].get("dt")
            )
            solution_data[start:stop] = solved.detach().cpu().numpy().reshape(batch_array.shape)
            if relative_stop % 50 == 0 or relative_stop == solved_count:
                print(f"new solver generation: {relative_stop}/{solved_count}", flush=True)
        for name, value in common_attrs.items():
            handle.attrs[name] = value
        handle.attrs["solver_seconds"] = time.perf_counter() - solver_started
        handle.attrs["device"] = str(device)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            handle.attrs["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
    if merge_existing:
        initial_backup = initial_path.with_suffix(initial_path.suffix + ".premerge")
        solution_backup = solution_path.with_suffix(solution_path.suffix + ".premerge")
        if initial_backup.exists() or solution_backup.exists():
            raise FileExistsError("premerge backup already exists; refusing to overwrite it")
        os.replace(initial_path, initial_backup)
        os.replace(solution_path, solution_backup)
        try:
            os.replace(initial_tmp, initial_path)
            os.replace(solution_tmp, solution_path)
        except Exception:
            if initial_path.exists():
                os.replace(initial_path, initial_tmp)
            os.replace(initial_backup, initial_path)
            os.replace(solution_backup, solution_path)
            raise
        print(f"Preserved pre-merge files as {initial_backup} and {solution_backup}", flush=True)
    else:
        os.replace(solution_tmp, solution_path)

    print(f"Direct generation seconds: {direct_seconds:.3f}", flush=True)
    print(f"Solver phase seconds: {time.perf_counter() - solver_started:.3f}", flush=True)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        print(f"Peak allocated GPU bytes: {torch.cuda.max_memory_allocated()}", flush=True)
    return initial_path, solution_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON configuration file")
    parser.add_argument(
        "--large-experiment", action="store_true",
        help="run the streaming experiment described by data_generation.md",
    )
    args = parser.parse_args()
    if args.large_experiment:
        print(generate_large_experiment(args.config))
    else:
        print(generate_dataset(args.config))


if __name__ == "__main__":
    main()
