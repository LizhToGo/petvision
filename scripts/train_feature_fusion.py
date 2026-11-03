"""Train a feature-level fusion head on frozen raw/guided ResNet features."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from petvision.config import RANDOM_SEED
from petvision.data import OxfordIIITPetIndex
from petvision.data.dataset import OxfordPetDataset
from petvision.data.splits import stratified_train_val
from petvision.data.transforms import PairedTransform
from petvision.evaluation import ClassificationMeter
from petvision.models import FeatureFusionHead, UNet, classifier_from_checkpoint
from petvision.training import apply_soft_mask
from petvision.training.utils import dataloader_options, save_json, set_seed


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_segmenter(path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_feature_model(
    path: Path,
    device: torch.device,
) -> tuple[nn.Module, int, float, tuple[str, ...]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture", "small_cnn") != "resnet18":
        raise SystemExit(f"Feature fusion requires a ResNet-18 checkpoint: {path}")
    model = classifier_from_checkpoint(checkpoint, device).eval()
    if not hasattr(model, "backbone"):
        raise TypeError(f"Checkpoint has no ResNet backbone: {path}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return (
        model,
        int(checkpoint["image_size"]),
        float(checkpoint.get("alpha", 0.25)),
        tuple(checkpoint["class_names"]),
    )


def _features(
    images: torch.Tensor,
    segmenter: UNet,
    raw_model: nn.Module,
    guided_model: nn.Module,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        raw_feature = raw_model.backbone(images)
        segmentation_logits = segmenter(images)
        guided_images = apply_soft_mask(images, segmentation_logits, alpha)
        guided_feature = guided_model.backbone(guided_images)
    return raw_feature, guided_feature


def _run_epoch(
    head: FeatureFusionHead,
    loader: DataLoader,
    segmenter: UNet,
    raw_model: nn.Module,
    guided_model: nn.Module,
    alpha: float,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, object]:
    training = optimizer is not None
    head.train(training)
    segmenter.eval()
    raw_model.eval()
    guided_model.eval()
    meter = ClassificationMeter(head.classifier[-1].out_features)
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        non_blocking = device.type == "cuda"
        images = batch["image"].to(device, non_blocking=non_blocking)
        labels = batch["label"].to(device, non_blocking=non_blocking)
        raw_feature, guided_feature = _features(
            images,
            segmenter,
            raw_model,
            guided_model,
            alpha,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = head(raw_feature, guided_feature)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = labels.shape[0]
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        meter.update(logits, labels)
    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(sample_count, 1)
    return metrics


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
        default=Path("artifacts/classification/resnet18/raw/best.pt"),
    )
    parser.add_argument(
        "--guided-checkpoint",
        type=Path,
        default=Path("artifacts/classification/resnet18/guided/best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature_fusion"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augmentation", choices=("none", "color"), default="none")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=not args.fast)
    device = _device(args.device)
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
    alpha = guided_alpha
    if abs(raw_alpha - guided_alpha) > 1e-8:
        raise SystemExit("Raw and guided checkpoints use different alpha values")

    index = OxfordIIITPetIndex(args.root)
    train_records, val_records = stratified_train_val(
        index.records("trainval"), args.val_fraction, args.seed
    )
    if args.limit > 0:
        train_records = train_records[: args.limit]
        val_records = val_records[: max(1, args.limit // 4)]
    train_transform = PairedTransform(
        image_size,
        train=args.augmentation == "color",
        color_jitter=args.augmentation == "color",
    )
    val_transform = PairedTransform(image_size, train=False)
    train_loader = DataLoader(
        OxfordPetDataset(index, train_records, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        **dataloader_options(device, args.num_workers),
    )
    val_loader = DataLoader(
        OxfordPetDataset(index, val_records, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_options(device, args.num_workers),
    )

    head = FeatureFusionHead(num_classes=index.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_f1 = -1.0
    history: list[dict[str, object]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        head.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_macro_f1", -1.0))
        history = list(checkpoint.get("history", []))

    print(
        f"device={device} feature_fusion train={len(train_records)} "
        f"val={len(val_records)} image_size={image_size} "
        f"augmentation={args.augmentation} start_epoch={start_epoch}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_started = time.perf_counter()
        train_metrics = _run_epoch(
            head,
            train_loader,
            segmentation,
            raw_model,
            guided_model,
            alpha,
            device,
            criterion,
            optimizer,
        )
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        with torch.no_grad():
            val_metrics = _run_epoch(
                head,
                val_loader,
                segmentation,
                raw_model,
                guided_model,
                alpha,
                device,
                criterion,
                None,
            )
        validation_seconds = time.perf_counter() - validation_started
        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = float(val_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "timing": {
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "train_samples_per_second": len(train_records)
                / max(train_seconds, 1e-9),
            },
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(row)
        checkpoint = {
            "model_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "image_size": image_size,
            "class_names": list(index.class_names),
            "architecture": "feature_fusion_head",
            "feature_dim": head.feature_dim,
            "alpha": alpha,
            "augmentation": args.augmentation,
            "raw_checkpoint": str(args.raw_checkpoint),
            "guided_checkpoint": str(args.guided_checkpoint),
            "segmentation_checkpoint": str(args.segmentation_checkpoint),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "fast": args.fast,
            "best_macro_f1": best_f1,
            "seed": args.seed,
            "history": history,
        }
        torch.save(checkpoint, output / "last.pt")
        if is_best:
            torch.save(checkpoint, output / "best.pt")
        save_json(output / "history.json", history)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"train_s={train_seconds:.1f}"
        )
    save_json(output / "dataset.json", index.summary())


if __name__ == "__main__":
    main()
