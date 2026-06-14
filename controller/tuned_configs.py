from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml

from .mppi import MPPIConfig


TUNED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "tuned_hyperparameters.yaml"
CONFIG_FIELD_NAMES = {field.name for field in fields(MPPIConfig)}


def load_tuned_config(
    robot: str,
    algo: str,
    *,
    path: Path = TUNED_CONFIG_PATH,
    overrides: dict[str, Any] | None = None,
) -> MPPIConfig:
    """Load a tuned MPPI config from the repository YAML file."""

    data = load_tuned_config_data(path)
    try:
        raw_config = data["dynamics"][robot]["algorithms"][algo]["config"]
    except KeyError as exc:
        raise KeyError(f"No tuned config found for robot={robot!r}, algo={algo!r} in {path}.") from exc
    if not isinstance(raw_config, dict):
        raise TypeError(f"Tuned config for robot={robot!r}, algo={algo!r} must be a mapping.")

    config_kwargs = {key: value for key, value in raw_config.items() if key in CONFIG_FIELD_NAMES}
    config = MPPIConfig(**config_kwargs)
    if overrides:
        unknown = sorted(set(overrides) - CONFIG_FIELD_NAMES)
        if unknown:
            raise KeyError(f"Unknown MPPIConfig override field(s): {unknown}")
        config = replace(config, **overrides)
    return config


def load_tuned_config_data(path: Path = TUNED_CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Tuned config file {path} must contain a mapping.")
    return data
