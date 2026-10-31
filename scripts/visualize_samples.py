"""Save a grid of Oxford-IIIT Pet images and segmentation annotations."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from petvision.data import OxfordIIITPetIndex, trimap_to_binary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_preview.png"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index = OxfordIIITPetIndex(args.root)
    records = list(index.records("trainval"))
    rng = random.Random(args.seed)
    selected = rng.sample(records, min(args.count, len(records)))
    columns = 3
    rows = len(selected)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    axes = np.atleast_2d(axes)

    for row, record in enumerate(selected):
        image, mask = index.load_images(record)
        binary = trimap_to_binary(mask)
        image_array = np.asarray(image)
        overlay = image_array.copy()
        overlay[binary == 1] = np.clip(
            0.7 * overlay[binary == 1] + np.array([0, 80, 0]),
            0,
            255,
        ).astype(np.uint8)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{record.image_id}\n{record.breed}")
        axes[row, 1].imshow(mask, cmap="viridis", vmin=1, vmax=2)
        axes[row, 1].set_title("foreground mask")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("foreground overlay")
        for column in range(columns):
            axes[row, column].axis("off")

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"saved {args.output} ({len(selected)} samples)")


if __name__ == "__main__":
    main()
