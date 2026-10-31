"""Plot JSON training histories for the report and presentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _series(history: list[dict[str, object]], split: str, key: str) -> list[float]:
    return [float(row[split][key]) for row in history]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("classification", "segmentation"),
        required=True,
    )
    args = parser.parse_args()

    history = _load(args.history)
    epochs = [int(row["epoch"]) for row in history]
    if args.kind == "classification":
        keys = (("loss", "Loss"), ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"))
    else:
        keys = (("loss", "Loss"), ("miou", "mIoU"), ("foreground_dice", "Foreground Dice"))

    figure, axes = plt.subplots(1, len(keys), figsize=(15, 4))
    for axis, (key, title) in zip(axes, keys):
        axis.plot(epochs, _series(history, "train", key), label="train")
        axis.plot(epochs, _series(history, "validation", key), label="validation")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
