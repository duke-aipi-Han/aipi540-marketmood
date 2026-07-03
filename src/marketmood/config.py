"""Configuration helpers for MarketMood."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Thin wrapper around the YAML configuration dictionary."""

    values: dict[str, Any]
    source_path: Path

    def get_path(self, key: str) -> Path:
        """Return a configured path from the top-level paths section."""
        return Path(self.values["paths"][key])


def load_config(config_path: str | Path = "config.yaml") -> ProjectConfig:
    """Load the project configuration from YAML."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file)
    return ProjectConfig(values=values, source_path=path)
