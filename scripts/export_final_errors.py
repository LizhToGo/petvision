"""Export high-confidence raw-classifier errors from the frozen test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from petvision.data import OxfordIIITPetIndex, OxfordPetDataset, PairedTransform
from petvision.data.transforms import image_to_tensor
from petvision.models import UNet, classifier_from_checkpoint
from petvision.training.utils import dataloader_options, set_seed


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_classifier(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return classifier_from_checkpoint(checkpoint, device).eval(), checkpoint


def _load_segmenter(path: Path, device: torch.device) -> tuple[UNet, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), int(checkpoint.get("image_size", 256))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_evaluation/errors"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    set_seed(args.seed)
    device = _device(args.device)
    index = OxfordIIITPetIndex(args.root)
    records = index.records("test")
    classifier, checkpoint = _load_classifier(args.raw_checkpoint, device)
    image_size = int(checkpoint["image_size"])
    dataset = OxfordPetDataset(index, records, PairedTransform(image_size, train=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_options(device, args.num_workers),
    )
    class_names = tuple(checkpoint["class_names"])
    errors: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            logits = classifier(images)
            probabilities = logits.softmax(dim=1)
            confidences, predictions = probabilities.max(dim=1)
            top_k = min(5, probabilities.shape[1])
            top_values, top_indices = probabilities.topk(top_k, dim=1)
            labels = batch["label"]
            for index_in_batch, image_id in enumerate(batch["image_id"]):
                true_index = int(labels[index_in_batch])
                predicted_index = int(predictions[index_in_batch])
                if true_index == predicted_index:
                    continue
                errors.append(
                    {
                        "image_id": str(image_id),
                        "true_breed": class_names[true_index],
                        "predicted_breed": class_names[predicted_index],
                        "confidence": float(confidences[index_in_batch]),
                        "top5": [
                            {
                                "breed": class_names[int(class_index)],
                                "probability": float(value),
                            }
                            for value, class_index in zip(
                                top_values[index_in_batch].cpu(),
                                top_indices[index_in_batch].cpu(),
                            )
                        ],
                    }
                )

    errors.sort(key=lambda item: float(item["confidence"]), reverse=True)
    selected = errors[: args.count]
    segmenter, segmentation_size = _load_segmenter(
        args.segmentation_checkpoint, device
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for rank, item in enumerate(selected, start=1):
        record = next(record for record in records if record.image_id == item["image_id"])
        image, _ = index.load_images(record)
        image_tensor, resized = image_to_tensor(image, segmentation_size)
        with torch.inference_mode():
            logits = segmenter(image_tensor.unsqueeze(0).to(device))
            mask = logits.softmax(dim=1)[0, 1].ge(0.5).cpu().numpy()
        rgb = np.asarray(resized, dtype=np.uint8)
        overlay = rgb.copy()
        overlay[mask] = (
            0.65 * overlay[mask] + np.array([0, 100, 0])
        ).astype(np.uint8)
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb)
        axes[0].set_title(
            f"original\ntrue: {item['true_breed']}"
        )
        axes[1].imshow(overlay)
        axes[1].set_title(
            f"predicted: {item['predicted_breed']}\n"
            f"confidence: {float(item['confidence']):.2%}"
        )
        axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("predicted foreground mask")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        filename = f"error_{rank:02d}_{record.image_id}.png"
        figure.savefig(args.output / filename, dpi=150)
        plt.close(figure)
        manifest.append({**item, "figure": filename})

    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "split": "test",
                "total_errors": len(errors),
                "selected_errors": manifest,
                "selection": "highest-confidence incorrect raw predictions",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {len(manifest)} errors from {len(errors)} total errors to {args.output}")


if __name__ == "__main__":
    main()
