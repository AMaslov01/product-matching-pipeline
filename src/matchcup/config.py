from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    raw_dir: Path
    work_dir: Path

    @property
    def canonical_dir(self) -> Path:
        return self.work_dir / "canonical"

    @property
    def pairs_dir(self) -> Path:
        return self.work_dir / "pairs"

    @property
    def models_dir(self) -> Path:
        return self.work_dir / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.work_dir / "outputs"


class Config:
    def __init__(self, values: dict[str, Any], source: Path) -> None:
        self.values = values
        self.source = source
        paths = values["paths"]
        # Collaborators keep raw data and all generated artifacts out of Git.
        # The checked-in recipe remains identical while private paths are bound
        # only at runtime through the ignored collaborator environment file.
        raw_dir = os.environ.get("MATCHCUP_RAW_DIR", paths["raw_dir"])
        work_dir = os.environ.get("MATCHCUP_WORK_DIR", paths["work_dir"])
        self.paths = ProjectPaths(Path(raw_dir), Path(work_dir))

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.values.get(name, {}))

    def get(self, dotted: str, default: Any = None) -> Any:
        value: Any = self.values
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


def load_config(path: str | Path) -> Config:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict) or "paths" not in values:
        raise ValueError(f"Invalid configuration: {source}")
    return Config(values, source)
