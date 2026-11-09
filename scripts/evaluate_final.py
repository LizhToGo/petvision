"""Evaluate the frozen final models once on the Oxford-IIIT Pet test split."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from petvision.config import IGNORE_INDEX, RANDOM_SEED
from petvision.data import OxfordIIITPetIndex, OxfordPetCropDataset, OxfordPetDataset, PairedTransform
from petvision.data.crops import load_crop_cache
from petvision.models import UNet, classifier_from_checkpoint
from petvision.training import run_classification_epoch, run_segmentation_epoch
from petvision.training.utils import dataloader_options, save_json, set_seed


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_segmentation(
    checkpoint_path: Path, device: torch.device
) -> tuple[UNet, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, int(checkpoint.get("image_size", 320))


def _load_classifier(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return classifier_from_checkpoint(checkpoint, device), checkpoint


def _loader(
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_options(device, num_workers),
    )


def _evaluate_classifier(
    checkpoint_path: Path,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, object], dict[str, object]]:
    model, checkpoint = _load_classifier(checkpoint_path, device)
    loader = _loader(dataset, device, batch_size, num_workers)
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    with torch.inference_mode():
        metrics = run_classification_epoch(model, loader, device, criterion)
    metrics["evaluation_seconds"] = time.perf_counter() - started
    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, {
        "path": str(checkpoint_path),
        "architecture": checkpoint.get("architecture"),
        "image_size": checkpoint.get("image_size"),
        "mode": checkpoint.get("mode"),
        "precision": checkpoint.get("precision", "fp32"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--cosine-checkpoint", type=Path, default=None)
    parser.add_argument("--crop-checkpoint", type=Path, default=None)
    parser.add_argument("--crop-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if (args.crop_checkpoint is None) != (args.crop_cache is None):
        parser.error("--crop-checkpoint and --crop-cache must be provided together")

    set_seed(args.seed)
    device = _device(args.device)
    index = OxfordIIITPetIndex(args.root)
    records = index.records("test")

    segmenter, segmentation_size = _load_segmentation(
        args.segmentation_checkpoint, device
    )
    segmentation_dataset = OxfordPetDataset(
        index, records, PairedTransform(segmentation_size, train=False)
    )
    segmentation_loader = _loader(
        segmentation_dataset, device, args.batch_size, args.num_workers
    )
    segmentation_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    started = time.perf_counter()
    with torch.inference_mode():
        segmentation_metrics = run_segmentation_epoch(
            segmenter,
            segmentation_loader,
            device,
            segmentation_criterion,
        )
    segmentation_metrics["evaluation_seconds"] = time.perf_counter() - started
    del segmenter, segmentation_loader, segmentation_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_state = torch.load(args.raw_checkpoint, map_location="cpu", weights_only=False)
    classifier_dataset = OxfordPetDataset(
        index,
        records,
        PairedTransform(int(raw_state["image_size"]), train=False),
    )
    classifiers: dict[str, dict[str, object]] = {}
    raw_metrics, raw_metadata = _evaluate_classifier(
        args.raw_checkpoint,
        classifier_dataset,
        device,
        args.batch_size,
        args.num_workers,
    )
    classifiers["raw_fp32"] = {"metrics": raw_metrics, "checkpoint": raw_metadata}

    if args.cosine_checkpoint is not None:
        cosine_metrics, cosine_metadata = _evaluate_classifier(
            args.cosine_checkpoint,
            classifier_dataset,
            device,
            args.batch_size,
            args.num_workers,
        )
        classifiers["raw_cosine"] = {
            "metrics": cosine_metrics,
            "checkpoint": cosine_metadata,
        }

    if args.crop_checkpoint is not None and args.crop_cache is not None:
        crop_metadata, crop_boxes = load_crop_cache(args.crop_cache)
        crop_state = torch.load(args.crop_checkpoint, map_location="cpu", weights_only=False)
        crop_dataset = OxfordPetCropDataset(
            index,
            records,
            PairedTransform(int(crop_state["image_size"]), train=False),
            crop_boxes,
        )
        crop_metrics, crop_checkpoint_metadata = _evaluate_classifier(
            args.crop_checkpoint,
            crop_dataset,
            device,
            args.batch_size,
            args.num_workers,
        )
        classifiers["predicted_crop_bf16"] = {
            "metrics": crop_metrics,
            "checkpoint": crop_checkpoint_metadata,
            "cache": {
                "path": str(args.crop_cache),
                "metadata": crop_metadata,
                "count": len(crop_boxes),
            },
        }

    result = {
        "dataset": index.summary(),
        "split": "test",
        "test_count": len(records),
        "seed": args.seed,
        "device": str(device),
        "segmentation": {
            "checkpoint": str(args.segmentation_checkpoint),
            "image_size": segmentation_size,
            "metrics": segmentation_metrics,
        },
        "classifiers": classifiers,
    }
    save_json(args.output / "metrics.json", result)
    print(
        "segmentation",
        "miou=%.4f" % segmentation_metrics["miou"],
        "foreground_iou=%.4f" % segmentation_metrics["foreground_iou"],
        "foreground_dice=%.4f" % segmentation_metrics["foreground_dice"],
    )
    for name, payload in classifiers.items():
        metrics = payload["metrics"]
        print(
            name,
            "accuracy=%.4f" % metrics["accuracy"],
            "macro_f1=%.4f" % metrics["macro_f1"],
            "top5=%.4f" % metrics["top5_accuracy"],
        )


if __name__ == "__main__":
    main()
