"""Lightweight YAML configuration helpers for the demo scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads a YAML file into a plain Python dictionary.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {config_path}")
    return data
