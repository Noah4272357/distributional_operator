"""Verify compact generation without any other project folders."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_generation_from_isolated_data_folder(tmp_path):
    isolated = tmp_path / "standalone_data"
    isolated.mkdir()
    shutil.copytree(
        ROOT / "data/isi_generation",
        isolated / "isi_generation",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(ROOT / "data/generate_isi_lif_laws.py", isolated)
    output = isolated / "generated/test.h5"
    command = [
        sys.executable,
        str(isolated / "generate_isi_lif_laws.py"),
        "--data_size", "24",
        "--sample_size", "16",
        "--n-isi", "16",
        "--dt", "0.004",
        "--t-max", "6",
        "--finite-bins", "16",
        "--output", "generated/test.h5",
    ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    with h5py.File(output, "r") as handle:
        assert handle.attrs["format"] == "isi_lif_laws"
        payload = {key: torch.from_numpy(handle[key][...]) for key in handle.keys()}
    assert set(payload) == {
        "law_ids",
        "regime_labels",
        "params",
        "normalized_params",
        "input_particles",
        "input_features",
        "bin_counts",
        "empirical_bin_mass",
        "bin_edges",
    }
    assert payload["input_particles"].shape == (24, 16)
    assert payload["normalized_params"].shape == (24, 2)
    assert payload["input_features"].shape == (24, 18)
    assert torch.allclose(payload["normalized_params"][:, 0], payload["params"][:, 0])
    assert torch.allclose(payload["normalized_params"][:, 1], payload["params"][:, 1].log())
    assert payload["bin_counts"].shape == (24, 17)
    assert torch.equal(payload["bin_counts"].sum(-1), torch.full((24,), 16.0))
    assert torch.allclose(payload["empirical_bin_mass"].sum(-1), torch.ones(24))

    second = isolated / "generated/second.h5"
    completed = subprocess.run(
        command[:-1] + ["generated/second.h5"],
        cwd=isolated,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    with h5py.File(second, "r") as handle:
        repeated = {key: torch.from_numpy(handle[key][...]) for key in handle.keys()}
    for key in payload:
        assert torch.equal(payload[key], repeated[key]), key
