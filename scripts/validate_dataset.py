"""Validate image/mask pairs and class coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from petvision.data import OxfordIIITPetIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="number of records to inspect; 0 means all records",
    )
    args = parser.parse_args()

    index = OxfordIIITPetIndex(args.root)
    records = index.records()
    selected = records if args.limit == 0 else records[: args.limit]
    counts: dict[int, int] = {}
    for record in selected:
        image, mask = index.load_images(record)
        if image.size != mask.size:
            raise SystemExit(
                f"size mismatch for {record.image_id}: "
                f"image={image.size} mask={mask.size}"
            )
        values = np.unique(np.asarray(mask)).tolist()
        invalid = set(values) - {1, 2, 3}
        if invalid:
            raise SystemExit(
                f"invalid mask values for {record.image_id}: {sorted(invalid)}"
            )
        counts[record.label] = counts.get(record.label, 0) + 1
    print(
        f"checked={len(selected)} total={len(records)} "
        f"classes={index.num_classes} labels_seen={len(counts)}"
    )
    if index.num_classes != 37:
        raise SystemExit(f"expected 37 classes, found {index.num_classes}")
    if len(counts) < min(index.num_classes, len(selected)):
        print("warning: selected prefix does not cover every class")


if __name__ == "__main__":
    main()
