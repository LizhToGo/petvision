"""Deterministic class-stratified train/validation splitting."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from petvision.data.oxford_pet import PetRecord


def stratified_train_val(
    records: Sequence[PetRecord],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[tuple[PetRecord, ...], tuple[PetRecord, ...]]:
    """Split records by label while preserving a deterministic result."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    groups: dict[int, list[PetRecord]] = defaultdict(list)
    for record in records:
        groups[record.label].append(record)

    rng = random.Random(seed)
    validation_ids: set[str] = set()
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        count = max(1, round(len(shuffled) * val_fraction))
        count = min(count, len(shuffled) - 1) if len(shuffled) > 1 else 0
        validation_ids.update(record.image_id for record in shuffled[:count])

    train = tuple(record for record in records if record.image_id not in validation_ids)
    validation = tuple(
        record for record in records if record.image_id in validation_ids
    )
    if not train or not validation:
        raise ValueError("The split produced an empty training or validation set")
    return train, validation
