"""Oxford-IIIT Pet indexing and foreground-mask conversion.

The conversion functions accept either the original 1/2/3 trimap or the
binary 1/2 masks prepared by the project data script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from PIL import Image

from petvision.config import (
    IGNORE_INDEX,
    TRIMAP_BACKGROUND,
    TRIMAP_BORDER,
    TRIMAP_FOREGROUND,
)

Split = Literal["trainval", "test"]


class DatasetLayoutError(RuntimeError):
    """Raised when the expected Oxford-IIIT Pet layout is missing."""


@dataclass(frozen=True)
class PetRecord:
    """One image, its foreground mask, and classification metadata."""

    image_id: str
    image_path: Path
    trimap_path: Path
    breed: str
    label: int
    split: Split


def _read_ids(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise DatasetLayoutError(f"Missing split annotation: {path}")

    image_ids: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image_id = line.split()[0]
        if image_id in image_ids:
            raise DatasetLayoutError(
                f"Duplicate image id {image_id!r} in {path}:{line_number}"
            )
        image_ids.append(image_id)
    return tuple(image_ids)


def _breed_from_image_id(image_id: str) -> str:
    try:
        breed, number = image_id.rsplit("_", 1)
    except ValueError as exc:
        raise DatasetLayoutError(
            f"Image id does not end with an image number: {image_id!r}"
        ) from exc
    if not breed or not number.isdigit():
        raise DatasetLayoutError(f"Invalid image id: {image_id!r}")
    return breed


def _require_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    image_dir = root / "images"
    annotation_dir = root / "annotations"
    trimap_dir = annotation_dir / "trimaps"
    trainval_file = annotation_dir / "trainval.txt"
    test_file = annotation_dir / "test.txt"
    required = (image_dir, trimap_dir, trainval_file, test_file)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\\n".join(missing)
        raise DatasetLayoutError(
            "Oxford-IIIT Pet dataset is incomplete. Missing:\n" + joined
        )
    return image_dir, trimap_dir, trainval_file, test_file


class OxfordIIITPetIndex:
    """Immutable index over the official trainval/test split files."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        image_dir, trimap_dir, trainval_file, test_file = _require_paths(
            self.root
        )
        split_ids: dict[Split, tuple[str, ...]] = {
            "trainval": _read_ids(trainval_file),
            "test": _read_ids(test_file),
        }
        all_ids = split_ids["trainval"] + split_ids["test"]
        if len(set(all_ids)) != len(all_ids):
            raise DatasetLayoutError("trainval.txt and test.txt overlap")

        breeds = tuple(sorted({_breed_from_image_id(image_id) for image_id in all_ids}))
        self.class_names = breeds
        self.class_to_idx = {breed: index for index, breed in enumerate(breeds)}
        self._records: dict[Split, tuple[PetRecord, ...]] = {}

        for split, ids in split_ids.items():
            records: list[PetRecord] = []
            for image_id in ids:
                breed = _breed_from_image_id(image_id)
                image_path = image_dir / f"{image_id}.jpg"
                trimap_path = trimap_dir / f"{image_id}.png"
                missing = [
                    str(path)
                    for path in (image_path, trimap_path)
                    if not path.is_file()
                ]
                if missing:
                    raise DatasetLayoutError(
                        f"Missing files for {image_id!r}: {', '.join(missing)}"
                    )
                records.append(
                    PetRecord(
                        image_id=image_id,
                        image_path=image_path,
                        trimap_path=trimap_path,
                        breed=breed,
                        label=self.class_to_idx[breed],
                        split=split,
                    )
                )
            self._records[split] = tuple(records)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def records(self, split: Split | None = None) -> tuple[PetRecord, ...]:
        """Return records in official annotation-file order."""

        if split is None:
            return self._records["trainval"] + self._records["test"]
        return self._records[split]

    def get(self, image_id: str) -> PetRecord:
        """Return a record by its dataset image id."""

        for record in self.records():
            if record.image_id == image_id:
                return record
        raise KeyError(image_id)

    def load_images(self, record: PetRecord) -> tuple[Image.Image, Image.Image]:
        """Load an RGB image and its grayscale mask or trimap."""

        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB")
        with Image.open(record.trimap_path) as trimap:
            gray_trimap = trimap.convert("L")
        return rgb, gray_trimap

    def summary(self) -> dict[str, object]:
        """Return a serializable dataset summary for logs and reports."""

        return {
            "root": str(self.root),
            "num_classes": self.num_classes,
            "classes": list(self.class_names),
            "trainval_count": len(self.records("trainval")),
            "test_count": len(self.records("test")),
        }


def _trimap_array(trimap: Image.Image | np.ndarray) -> np.ndarray:
    raw = np.asarray(trimap, dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2-D trimap, got shape {raw.shape}")
    values = set(np.unique(raw).tolist())
    allowed = {
        TRIMAP_FOREGROUND,
        TRIMAP_BACKGROUND,
        TRIMAP_BORDER,
    }
    invalid = values - allowed
    if invalid:
        raise ValueError(f"Unexpected trimap values: {sorted(invalid)}")
    return raw


def trimap_to_target(
    trimap: Image.Image | np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Convert a binary mask or trimap to a 0/1 target."""

    raw = _trimap_array(trimap)
    target = np.full(raw.shape, ignore_index, dtype=np.uint8)
    target[raw == TRIMAP_BACKGROUND] = 0
    target[raw == TRIMAP_FOREGROUND] = 1
    return target


def trimap_to_binary(trimap: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a binary mask or trimap to a display foreground mask."""

    raw = _trimap_array(trimap)
    return np.logical_or(
        raw == TRIMAP_FOREGROUND,
        raw == TRIMAP_BORDER,
    ).astype(np.uint8)


def records_by_split(
    records: Iterable[PetRecord],
) -> dict[Split, tuple[PetRecord, ...]]:
    """Group records by split without changing their order."""

    grouped: dict[Split, list[PetRecord]] = {"trainval": [], "test": []}
    for record in records:
        grouped[record.split].append(record)
    return {split: tuple(items) for split, items in grouped.items()}
