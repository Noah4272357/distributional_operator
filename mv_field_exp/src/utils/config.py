"""Plain-Python YAML configuration loading and dot-list overrides."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


class Config(dict[str, Any]):
    """Dictionary with recursive attribute access for concise configuration use."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _convert(value)


def _convert(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config({str(key): _convert(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_convert(item) for item in value]
    return value


def to_plain(value: Any) -> Any:
    """Convert nested Config objects into YAML/JSON-safe built-in containers."""
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """Apply ``section.key=value`` assignments parsed with YAML scalar rules."""
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"override must have KEY=VALUE form: {expression!r}")
        dotted_key, raw_value = expression.split("=", 1)
        keys = dotted_key.split(".")
        if not all(keys):
            raise ValueError(f"invalid override key: {dotted_key!r}")
        target: Config = cfg
        for key in keys[:-1]:
            current = target.get(key)
            if current is None:
                current = Config()
                target[key] = current
            if not isinstance(current, Config):
                raise ValueError(f"cannot set nested key below non-mapping: {key}")
            target = current
        target[keys[-1]] = _convert(yaml.safe_load(raw_value))
    return cfg


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    """Load YAML into a recursive Config and apply optional dot-list overrides."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"configuration root must be a mapping: {path}")
    cfg = _convert(payload)
    apply_overrides(cfg, overrides)
    if "experiment" in cfg:
        for key in ("experiment", "generated_data", "model", "loss", "optimizer", "random_field_view"):
            if key not in cfg:
                raise ValueError(f"missing required configuration section: {key}")
    return cfg


def save_config(cfg: Mapping[str, Any], run_dir: str | Path) -> None:
    """Save the resolved plain-YAML configuration in the run directory."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain(cfg), handle, sort_keys=False)
