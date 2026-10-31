"""Shared paths and immutable experiment defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the project."""

    project_root: Path
    data_root: Path
    dataset_root: Path
    artifact_root: Path


def default_paths() -> ProjectPaths:
    """Return paths, allowing the dataset root to be overridden by an env var."""

    data_root = Path(
        os.environ.get("PETVISION_DATA_ROOT", str(PROJECT_ROOT / "data"))
    ).expanduser()
    return ProjectPaths(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        dataset_root=data_root / "oxford-iiit-pet",
        artifact_root=PROJECT_ROOT / "artifacts",
    )


IMAGE_SIZE = 256
NUM_CLASSES = 37
RANDOM_SEED = 42
TRIMAP_FOREGROUND = 1
TRIMAP_BACKGROUND = 2
TRIMAP_BORDER = 3
IGNORE_INDEX = 255
