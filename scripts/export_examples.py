"""Export representative test-set predictions for the report."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from petvision.data import OxfordIIITPetIndex, trimap_to_binary
from petvision.inference import PetVisionInference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        default=Path("artifacts/segmentation/best.pt"),
    )
    parser.add_argument(
        "--raw-checkpoint",
        type=Path,
        default=Path("artifacts/classification/raw/best.pt"),
    )
    parser.add_argument(
        "--guided-checkpoint",
        type=Path,
        default=Path("artifacts/classification/guided/best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    index = OxfordIIITPetIndex(args.root)
    service = PetVisionInference.from_checkpoints(
        args.segmentation_checkpoint,
        args.raw_checkpoint,
        args.guided_checkpoint,
        args.device,
    )
    rng = random.Random(args.seed)
    records = list(index.records("test"))
    selected = rng.sample(records, min(args.count, len(records)))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for record in selected:
        image, trimap = index.load_images(record)
        prediction = service.predict(image)
        resized = image.resize(
            (service.image_size, service.image_size),
            Image.Resampling.BILINEAR,
        )
        gt_mask = trimap.resize(
            (service.image_size, service.image_size),
            Image.Resampling.NEAREST,
        )
        gt_binary = trimap_to_binary(gt_mask)
        raw_name, raw_probability = prediction.raw_top5[0]
        guided_name, guided_probability = prediction.guided_top5[0]
        manifest.append(
            {
                "image_id": record.image_id,
                "true_breed": record.breed,
                "raw_prediction": raw_name,
                "guided_prediction": guided_name,
                "raw_probability": raw_probability,
                "guided_probability": guided_probability,
                "raw_correct": raw_name == record.breed,
                "guided_correct": guided_name == record.breed,
            }
        )

        figure, axes = plt.subplots(2, 3, figsize=(13, 8))
        axes[0, 0].imshow(resized)
        axes[0, 0].set_title(f"original\ntrue: {record.breed}")
        axes[0, 1].imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title("ground-truth binary mask")
        axes[0, 2].imshow(prediction.mask, cmap="gray", vmin=0, vmax=1)
        axes[0, 2].set_title("predicted mask")
        axes[1, 0].imshow(prediction.overlay)
        axes[1, 0].set_title("overlay")
        axes[1, 1].imshow(prediction.foreground_image)
        axes[1, 1].set_title("foreground extraction")
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.02,
            0.95,
            f"raw: {raw_name} ({raw_probability:.2%})\n"
            f"guided: {guided_name} ({guided_probability:.2%})",
            va="top",
            ha="left",
            wrap=True,
        )
        for axis in axes.flat:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(args.output / f"{record.image_id}.png", dpi=150)
        plt.close(figure)

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {len(manifest)} examples to {args.output}")


if __name__ == "__main__":
    main()
