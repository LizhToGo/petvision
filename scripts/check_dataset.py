"""Print a concise Oxford-IIIT Pet dataset summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from petvision.data import OxfordIIITPetIndex
from petvision.data.oxford_pet import DatasetLayoutError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/oxford-iiit-pet"),
    )
    args = parser.parse_args()

    try:
        index = OxfordIIITPetIndex(args.root)
    except DatasetLayoutError as exc:
        raise SystemExit(str(exc)) from exc

    summary = index.summary()
    print(f"root: {summary['root']}")
    print(f"classes: {summary['num_classes']}")
    print(f"trainval: {summary['trainval_count']}")
    print(f"test: {summary['test_count']}")
    print("first classes:", ", ".join(index.class_names[:5]))


if __name__ == "__main__":
    main()
