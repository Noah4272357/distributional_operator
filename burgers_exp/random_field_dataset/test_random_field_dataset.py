"""Tests and requested sample visualizations for the random-field package."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from random_field_dataset.burgers_solver import solve_burgers
from random_field_dataset.generate_dataset import generate_dataset
from random_field_dataset.generate_dataset import _parameter_grid, generate_large_experiment
from random_field_dataset.generate_initial_condition import generate_initial_condition


OUTPUT_DIR = Path(__file__).parent / "test_outputs"


def _plot(fields: np.ndarray, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 5, figsize=(13, 5), constrained_layout=True)
    for axis, field in zip(axes.flat, fields):
        axis.plot(field)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def test_processes_and_solver_visualizations() -> None:
    for process_type in ("truncated_Gaussian", "student_t"):
        kwargs = {"nu": 4.0} if process_type == "student_t" else {}
        initial = generate_initial_condition(
            10, Nx=64, process_type=process_type, seed=7,
            trunc_dim=8, sigma=1.0, alpha=2.0, s=2.0, **kwargs
        )
        assert initial.shape == (10, 64)
        _plot(initial, OUTPUT_DIR / f"{process_type}_sample.png")
        solution = solve_burgers(initial, visc=0.02, T=0.02)
        assert solution.shape == initial.shape
        assert np.isfinite(solution).all()
        _plot(solution, OUTPUT_DIR / f"{process_type}_solution.png")


def test_dirichlet_boundaries() -> None:
    initial = generate_initial_condition(
        10, Nx=64, boundary_type="dirichlet", seed=9,
        trunc_dim=8, sigma=1.0, alpha=2.0, s=2.0
    )
    solution = solve_burgers(initial, visc=0.02, T=0.02, boundary_type="dirichlet")
    assert np.all(initial[:, (0, -1)] == 0)
    assert np.all(solution[:, (0, -1)] == 0)


def test_configured_dataset(tmp_path: Path) -> None:
    base = json.loads((Path(__file__).parent / "config.json").read_text())
    base["initial_condition"]["Nx"] = 64
    base["N_process"] = 8
    base["solver"]["T"] = 0.01
    base["output_path"] = str(tmp_path / "dataset.h5")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(base))
    output = generate_dataset(config_path)
    with h5py.File(output, "r") as handle:
        expected = 8 * base["initial_condition"]["N_sample"]
        assert handle["initial"].shape == (expected, 64)
        assert handle["solution"].shape == (expected, 64)
        parameters = json.loads(handle.attrs["process_parameters"])
        assert len(parameters) == 8


def test_large_parameter_grid() -> None:
    config = {
        "parameter_values_per_arg": 2,
        "process_args_range": {
            "trunc_dim": 8,
            "sigma": {"distribution": "uniform", "range": [0.5, 1.5]},
            "alpha": {"distribution": "loguniform", "range": [1.0, 10.0]},
            "s": {"distribution": "uniform", "range": [1.5, 4.0]},
        },
    }
    grid = _parameter_grid(config, np.random.default_rng(1))
    assert len(grid) == 8
    assert all(item["trunc_dim"] == 8 for item in grid)


def test_small_streaming_experiment(tmp_path: Path) -> None:
    config = {
        "seed": 1,
        "parameter_values_per_arg": 2,
        "process_args_range": {
            "trunc_dim": 3,
            "sigma": {"distribution": "uniform", "range": [0.5, 1.5]},
            "alpha": {"distribution": "loguniform", "range": [1.0, 10.0]},
            "s": {"distribution": "uniform", "range": [1.5, 4.0]},
        },
        "process_type": "student_t",
        "student_t_nu": 4.0,
        "direct_processes": 4,
        "solved_processes": 4,
        "N_sample": 3,
        "Nx": 32,
        "domain": [-1.0, 1.0],
        "boundary_type": "periodic",
        "solver_batch_processes": 2,
        "device": "cpu",
        "solver": {"visc": 0.01, "T": 0.01, "dt": None},
        "output_dir": str(tmp_path / "output"),
    }
    config_path = tmp_path / "large.json"
    config_path.write_text(json.dumps(config))
    initial_path, solution_path = generate_large_experiment(config_path)
    with h5py.File(initial_path, "r") as handle:
        assert handle["initial"].shape == (4, 3, 32)
        assert handle.attrs["process_type"] == "student_t"
        initial_indices = set(handle["parameter_index"][:])
    with h5py.File(solution_path, "r") as handle:
        assert handle["solution"].shape == (4, 3, 32)
        assert "initial" not in handle
        assert handle.attrs["process_type"] == "student_t"
        assert np.isfinite(handle["solution"][:]).all()
        solution_indices = set(handle["parameter_index"][:])
        assert initial_indices.isdisjoint(solution_indices)
        assert initial_indices | solution_indices == set(range(8))


def test_merge_streaming_experiment(tmp_path: Path) -> None:
    config = {
        "seed": 11,
        "parameter_values_per_arg": 2,
        "process_args_range": {
            "trunc_dim": 3,
            "sigma": {"distribution": "uniform", "range": [0.5, 1.5]},
            "alpha": {"distribution": "loguniform", "range": [1.0, 10.0]},
            "s": {"distribution": "uniform", "range": [1.5, 4.0]},
        },
        "process_type": "student_t",
        "student_t_nu": 4.0,
        "direct_processes": 4,
        "solved_processes": 4,
        "N_sample": 2,
        "Nx": 16,
        "domain": [-1.0, 1.0],
        "boundary_type": "periodic",
        "solver_batch_processes": 2,
        "device": "cpu",
        "solver": {"visc": 0.01, "T": 0.005, "dt": None},
        "output_dir": str(tmp_path / "output"),
    }
    config_path = tmp_path / "merge.json"
    config_path.write_text(json.dumps(config))
    initial_path, solution_path = generate_large_experiment(config_path)
    with h5py.File(initial_path, "r") as handle:
        original_initial = handle["initial"][:]
    with h5py.File(solution_path, "r") as handle:
        original_solution = handle["solution"][:]

    config.update({"seed": 12, "merge_existing": True, "target_processes_per_file": 8})
    config_path.write_text(json.dumps(config))
    generate_large_experiment(config_path)
    with h5py.File(initial_path, "r") as handle:
        assert handle["initial"].shape == (8, 2, 16)
        assert np.array_equal(handle["initial"][:4], original_initial)
        assert list(handle["generation_id"][:]) == [0] * 4 + [1] * 4
        assert len(json.loads(handle.attrs["parameter_grids"])) == 2
    with h5py.File(solution_path, "r") as handle:
        assert handle["solution"].shape == (8, 2, 16)
        assert np.array_equal(handle["solution"][:4], original_solution)
        assert list(handle["generation_id"][:]) == [0] * 4 + [1] * 4
        assert len(json.loads(handle.attrs["configs"])) == 2
    assert initial_path.with_suffix(".hdf5.premerge").exists()
    assert solution_path.with_suffix(".hdf5.premerge").exists()


def test_sample_split_experiment(tmp_path: Path) -> None:
    config = {
        "seed": 21,
        "parameter_values_per_arg": {"sigma": 3, "alpha": 2, "s": 2},
        "process_args_range": {
            "trunc_dim": 3,
            "sigma": {"distribution": "uniform", "range": [0.5, 1.5]},
            "alpha": {"distribution": "loguniform", "range": [1.0, 10.0]},
            "s": {"distribution": "uniform", "range": [1.5, 4.0]},
        },
        "process_type": "student_t",
        "student_t_nu": 4.0,
        "split_samples": True,
        "N_sample": 4,
        "stored_samples_per_process": 2,
        "Nx": 16,
        "domain": [-1.0, 1.0],
        "boundary_type": "periodic",
        "solver_batch_processes": 2,
        "device": "cpu",
        "solver": {"visc": 0.01, "T": 0.005, "dt": None},
        "output_dir": str(tmp_path / "sample_split"),
    }
    config_path = tmp_path / "sample_split.json"
    config_path.write_text(json.dumps(config))
    initial_path, solution_path = generate_large_experiment(config_path)
    with h5py.File(initial_path, "r") as initial_handle, h5py.File(
        solution_path, "r"
    ) as solution_handle:
        assert initial_handle["initial"].shape == (12, 2, 16)
        assert solution_handle["solution"].shape == (12, 2, 16)
        assert "solution" not in initial_handle
        assert "initial" not in solution_handle
        assert np.array_equal(
            initial_handle["parameter_index"][:], solution_handle["parameter_index"][:]
        )
        assert np.array_equal(
            initial_handle["sample_seed"][:], solution_handle["sample_seed"][:]
        )
        assert np.isfinite(initial_handle["initial"][:]).all()
        assert np.isfinite(solution_handle["solution"][:]).all()
