"""Build normalized pet crop boxes from predicted or oracle masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from petvision.data import OxfordIIITPetIndex, OxfordPetDataset, PairedTransform
from petvision.models import UNet
from petvision.training.utils import dataloader_options, save_json, set_seed


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _box_from_mask(mask: torch.Tensor, margin: float) -> tuple[list[float], bool]:
    height, width = mask.shape
    coordinates = torch.where(mask)
    if coordinates[0].numel() == 0:
        return [0.0, 0.0, 1.0, 1.0], True
    top = float(coordinates[0].min().item()) / height
    bottom = float(coordinates[0].max().item() + 1) / height
    left = float(coordinates[1].min().item()) / width
    right = float(coordinates[1].max().item() + 1) / width
    box_width = right - left
    box_height = bottom - top
    return [
        max(0.0, left - margin * box_width),
        max(0.0, top - margin * box_height),
        min(1.0, right + margin * box_width),
        min(1.0, bottom + margin * box_height),
    ], False


def _load_segmenter(path: Path, device: torch.device) -> tuple[UNet, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, int(checkpoint.get("image_size", 256))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("predicted", "oracle"), default="predicted")
    parser.add_argument("--segmentation-checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("trainval", "test", "all"), default="trainval")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.margin < 0.0:
        parser.error("--margin must be non-negative")
    if args.source == "predicted" and args.segmentation_checkpoint is None:
        parser.error("predicted source requires --segmentation-checkpoint")

    set_seed(args.seed)
    device = _device(args.device)
    index = OxfordIIITPetIndex(args.root)
    records = index.records(args.split) if args.split != "all" else index.records()
    if args.limit > 0:
        records = records[: args.limit]
    transform_size = args.image_size
    segmenter = None
    checkpoint_image_size = None
    if args.source == "predicted":
        segmenter, checkpoint_image_size = _load_segmenter(
            args.segmentation_checkpoint, device
        )
        transform_size = checkpoint_image_size

    dataset = OxfordPetDataset(index, records, PairedTransform(transform_size, train=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_options(device, args.num_workers),
    )
    boxes: dict[str, list[float]] = {}
    fallback_count = 0
    foreground_fractions: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            if args.source == "predicted":
                images = batch["image"].to(device, non_blocking=device.type == "cuda")
                probabilities = segmenter(images).softmax(dim=1)[:, 1]
                masks = probabilities >= args.threshold
            else:
                masks = (batch["mask"] == 1) | (batch["mask"] == 255)
            for image_id, mask in zip(batch["image_id"], masks):
                box, fallback = _box_from_mask(mask.cpu(), args.margin)
                boxes[str(image_id)] = box
                fallback_count += int(fallback)
                foreground_fractions.append(float(mask.float().mean().item()))

    metadata = {
        "root": str(index.root),
        "source": args.source,
        "split": args.split,
        "count": len(boxes),
        "image_size": transform_size,
        "threshold": args.threshold if args.source == "predicted" else None,
        "margin": args.margin,
        "segmentation_checkpoint": (
            str(args.segmentation_checkpoint)
            if args.source == "predicted"
            else None
        ),
        "seed": args.seed,
        "fallback_count": fallback_count,
        "mean_foreground_fraction": sum(foreground_fractions) / max(len(foreground_fractions), 1),
    }
    save_json(args.output, {"metadata": metadata, "boxes": boxes})
    print(
        f"source={args.source} split={args.split} boxes={len(boxes)} "
        f"fallback={fallback_count} output={args.output}"
    )


if __name__ == "__main__":
    main()
