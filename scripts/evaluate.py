"""Evaluate segmentation and the three classification input conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from petvision.config import IGNORE_INDEX, RANDOM_SEED
from petvision.data import OxfordIIITPetIndex
from petvision.data.dataset import OxfordPetDataset
from petvision.data.transforms import PairedTransform
from petvision.models import SmallCNN, UNet
from petvision.training import run_classification_epoch, run_segmentation_epoch
from petvision.training.utils import save_json, set_seed


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_models(
    segmentation_checkpoint: Path,
    raw_checkpoint: Path,
    guided_checkpoint: Path,
    device: torch.device,
) -> tuple[UNet, SmallCNN, SmallCNN, dict[str, object]]:
    segmentation_state = torch.load(
        segmentation_checkpoint,
        map_location=device,
        weights_only=False,
    )
    raw_state = torch.load(raw_checkpoint, map_location=device, weights_only=False)
    guided_state = torch.load(
        guided_checkpoint,
        map_location=device,
        weights_only=False,
    )
    class_names = tuple(raw_state["class_names"])
    segmenter = UNet().to(device)
    segmenter.load_state_dict(segmentation_state["model_state"])
    raw_model = SmallCNN(num_classes=len(class_names)).to(device)
    raw_model.load_state_dict(raw_state["model_state"])
    guided_model = SmallCNN(num_classes=len(class_names)).to(device)
    guided_model.load_state_dict(guided_state["model_state"])
    metadata = {
        "class_names": list(class_names),
        "image_size": raw_state["image_size"],
        "alpha": guided_state.get("alpha", 0.25),
    }
    return segmenter, raw_model, guided_model, metadata


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit test records for a smoke run",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = _device(args.device)
    index = OxfordIIITPetIndex(args.root)
    segmenter, raw_model, guided_model, metadata = _load_models(
        args.segmentation_checkpoint,
        args.raw_checkpoint,
        args.guided_checkpoint,
        device,
    )
    records = index.records("test")
    if args.limit > 0:
        records = records[: args.limit]
    dataset = OxfordPetDataset(
        index,
        records,
        PairedTransform(int(metadata["image_size"]), train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion_segmentation = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_classification = nn.CrossEntropyLoss()
    with torch.no_grad():
        segmentation = run_segmentation_epoch(
            segmenter,
            loader,
            device,
            criterion_segmentation,
        )
        raw = run_classification_epoch(
            raw_model,
            loader,
            device,
            criterion_classification,
        )
        guided = run_classification_epoch(
            guided_model,
            loader,
            device,
            criterion_classification,
            segmenter=segmenter,
            alpha=float(metadata["alpha"]),
        )
        oracle = run_classification_epoch(
            guided_model,
            loader,
            device,
            criterion_classification,
            alpha=float(metadata["alpha"]),
            oracle_mask=True,
        )

    result = {
        "dataset": index.summary(),
        "segmentation": segmentation,
        "classification_raw": raw,
        "classification_predicted_mask": guided,
        "classification_true_mask_oracle": oracle,
        "metadata": metadata,
    }
    save_json(args.output / "metrics.json", result)
    print(
        "segmentation",
        "loss=%.4f" % segmentation["loss"],
        "miou=%.4f" % segmentation["miou"],
        "foreground_iou=%.4f" % segmentation["foreground_iou"],
        "foreground_dice=%.4f" % segmentation["foreground_dice"],
    )
    for name, metrics in (
        ("raw", raw),
        ("predicted_mask", guided),
        ("true_mask_oracle", oracle),
    ):
        print(
            name,
            "loss=%.4f" % metrics["loss"],
            "accuracy=%.4f" % metrics["accuracy"],
            "macro_f1=%.4f" % metrics["macro_f1"],
            "top5=%.4f" % metrics["top5_accuracy"],
        )


if __name__ == "__main__":
    main()
