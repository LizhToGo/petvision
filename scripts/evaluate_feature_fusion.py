"""Evaluate a trained feature-level raw/guided fusion head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from petvision.data import OxfordIIITPetIndex
from petvision.data.dataset import OxfordPetDataset
from petvision.data.transforms import PairedTransform
from petvision.models import FeatureFusionHead
from petvision.training.utils import dataloader_options, save_json, set_seed

if __package__:
    from .train_feature_fusion import (
        _device,
        _load_feature_model,
        _load_segmenter,
        _run_epoch,
    )
else:
    from train_feature_fusion import (
        _device,
        _load_feature_model,
        _load_segmenter,
        _run_epoch,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--guided-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature_fusion_evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = _device(args.device)
    head_checkpoint = torch.load(
        args.fusion_checkpoint,
        map_location=device,
        weights_only=False,
    )
    segmentation = _load_segmenter(args.segmentation_checkpoint, device)
    raw_model, image_size, raw_alpha, raw_classes = _load_feature_model(
        args.raw_checkpoint,
        device,
    )
    guided_model, guided_image_size, guided_alpha, guided_classes = (
        _load_feature_model(args.guided_checkpoint, device)
    )
    if image_size != guided_image_size:
        raise SystemExit("Raw and guided checkpoints use different image sizes")
    if raw_classes != guided_classes:
        raise SystemExit("Raw and guided checkpoints use different classes")
    if abs(raw_alpha - guided_alpha) > 1e-8:
        raise SystemExit("Raw and guided checkpoints use different alpha values")
    if tuple(head_checkpoint["class_names"]) != raw_classes:
        raise SystemExit("Fusion head and classifier checkpoints use different classes")

    index = OxfordIIITPetIndex(args.root)
    records = index.records("test")
    if args.limit > 0:
        records = records[: args.limit]
    dataset = OxfordPetDataset(
        index,
        records,
        PairedTransform(image_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_options(device, args.num_workers),
    )
    head = FeatureFusionHead(
        num_classes=len(raw_classes),
        feature_dim=int(head_checkpoint.get("feature_dim", 512)),
    ).to(device)
    head.load_state_dict(head_checkpoint["model_state"])
    criterion = torch.nn.CrossEntropyLoss()
    with torch.inference_mode():
        metrics = _run_epoch(
            head,
            loader,
            segmentation,
            raw_model,
            guided_model,
            guided_alpha,
            device,
            criterion,
            None,
        )
    result = {
        "dataset": index.summary(),
        "metadata": {
            "image_size": image_size,
            "alpha": guided_alpha,
            "feature_dim": int(head_checkpoint.get("feature_dim", 512)),
            "raw_checkpoint": str(args.raw_checkpoint),
            "guided_checkpoint": str(args.guided_checkpoint),
            "fusion_checkpoint": str(args.fusion_checkpoint),
        },
        "test": metrics,
    }
    save_json(args.output / "metrics.json", result)
    print(
        "feature_fusion",
        "accuracy=%.4f" % metrics["accuracy"],
        "macro_f1=%.4f" % metrics["macro_f1"],
        "top5=%.4f" % metrics["top5_accuracy"],
    )


if __name__ == "__main__":
    main()
