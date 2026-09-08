from pathlib import Path
import subprocess
import sys

import h5py

from dataset.mckean_vlasov_generation import generate_mckean_vlasov_dataset
from src.utils.config import load_config


def test_default_config_is_ten_epoch_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "config.yaml")
    assert cfg.experiment.epochs == 10
    assert str(cfg.generated_data.path).startswith("dataset/generated/")
    assert (root / "scripts" / "visualize_random_field_covariance.py").exists()


def test_hdf5_generation_shape(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "generate.yaml")
    cfg.output_dir = str(tmp_path)
    cfg.filename = "tiny.h5"
    result = generate_mckean_vlasov_dataset(cfg, data_size=12, sample_size=5, grid_size=9)
    assert result["input_field_particles_shape"] == (12, 5, 9)
    with h5py.File(result["hdf5_path"], "r") as handle:
        assert handle["input_field_particles"].shape == (12, 5, 9)
        assert handle.attrs["target_name"] == "McKean_Vlasov"


def test_generate_cli_size_arguments(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "generate.py"),
            "--config",
            str(root / "configs" / "generate.yaml"),
            "--data_size",
            "15",
            "--sample_size",
            "7",
            "--grid_size",
            "11",
            f"output_dir={tmp_path}",
            "filename=cli.h5",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    with h5py.File(tmp_path / "cli.h5", "r") as handle:
        assert handle["input_field_particles"].shape == (15, 7, 11)
