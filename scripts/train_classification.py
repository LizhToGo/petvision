"""Train raw or segmentation-guided pet classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from petvision.config import IMAGE_SIZE, RANDOM_SEED
from petvision.data import OxfordIIITPetIndex
from petvision.data.dataset import OxfordPetDataset
from petvision.data.splits import stratified_train_val
from petvision.data.transforms import PairedTransform
from petvision.models import build_classifier
from petvision.models.unet import UNet
from petvision.training.loops import run_classification_epoch
from petvision.training.utils import save_json, set_seed


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_resume(
    checkpoint: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, list[dict[str, object]]]:
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    else:
        print("resume checkpoint has no optimizer state; restarting optimizer")
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_f1 = float(checkpoint.get("best_macro_f1", -1.0))
    history = list(checkpoint.get("history", []))
    return start_epoch, best_f1, history

def _load_segmenter(path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("raw", "guided"), default="raw")
    parser.add_argument("--architecture", choices=("small_cnn", "resnet18"), default=None)
    parser.add_argument(
        "--pretrained", action="store_true", help="initialize ResNet-18 with ImageNet-1K V1 weights"
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        type=Path,
        default=Path("artifacts/segmentation/best.pt"),
    )
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit train and validation records for a smoke run",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume from a last.pt or compatible checkpoint",
    )
    args = parser.parse_args()

    if args.mode == "guided" and not args.segmenter_checkpoint.is_file():
        raise SystemExit(
            f"Guided mode requires a segmentation checkpoint: "
            f"{args.segmenter_checkpoint}"
        )
    set_seed(args.seed)
    device = _device(args.device)
    resume_checkpoint = (
        torch.load(args.resume, map_location=device, weights_only=False)
        if args.resume is not None else None
    )
    checkpoint_architecture = (
        resume_checkpoint.get("architecture", "small_cnn")
        if resume_checkpoint is not None else None
    )
    if args.architecture is not None and checkpoint_architecture is not None:
        if args.architecture != checkpoint_architecture:
            raise SystemExit("--architecture does not match the resume checkpoint")
    architecture = args.architecture or checkpoint_architecture or "small_cnn"
    if args.pretrained and architecture != "resnet18":
        raise SystemExit("--pretrained is only supported for resnet18")
    pretrained_source = (
        resume_checkpoint.get("pretrained_source")
        if resume_checkpoint is not None
        else "torchvision:ResNet18_Weights.IMAGENET1K_V1" if args.pretrained else None
    )
    index = OxfordIIITPetIndex(args.root)
    train_records, val_records = stratified_train_val(
        index.records("trainval"), args.val_fraction, args.seed
    )
    if args.limit > 0:
        train_records = train_records[: args.limit]
        val_records = val_records[: max(1, args.limit // 4)]
    transform_train = PairedTransform(args.image_size, train=True)
    transform_val = PairedTransform(args.image_size, train=False)
    train_dataset = OxfordPetDataset(index, train_records, transform_train)
    val_dataset = OxfordPetDataset(index, val_records, transform_val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    segmenter = None
    if args.mode == "guided":
        segmenter = _load_segmenter(args.segmenter_checkpoint, device)
    if architecture == "resnet18":
        # Segmenter construction must not change the classifier's initialization.
        set_seed(args.seed)
    model = build_classifier(
        num_classes=index.num_classes,
        architecture=architecture,
        pretrained=args.pretrained and resume_checkpoint is None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    output = args.output or (
        Path("artifacts/classification/raw") if architecture == "small_cnn"
        else Path("artifacts/classification") / architecture / args.mode
    )
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_f1 = -1.0
    history: list[dict[str, object]] = []
    if resume_checkpoint is not None:
        start_epoch, best_f1, history = _load_resume(
            resume_checkpoint, model, optimizer
        )
        print(f"resuming from {args.resume} at epoch {start_epoch}")
    if start_epoch > args.epochs:
        raise SystemExit(
            f"checkpoint is already past requested epochs: "
            f"start={start_epoch} epochs={args.epochs}"
        )

    print(
        f"device={device} mode={args.mode} architecture={architecture} "
        f"train={len(train_dataset)} val={len(val_dataset)} "
        f"start_epoch={start_epoch}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_classification_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            segmenter,
            args.alpha,
        )
        with torch.no_grad():
            val_metrics = run_classification_epoch(
                model,
                val_loader,
                device,
                criterion,
                None,
                segmenter,
                args.alpha,
            )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(row)
        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = float(val_metrics["macro_f1"])
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "image_size": args.image_size,
            "class_names": list(index.class_names),
            "mode": args.mode,
            "architecture": architecture,
            "pretrained_source": pretrained_source,
            "alpha": args.alpha,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_fraction": args.val_fraction,
            "batch_size": args.batch_size,
            "segmenter_checkpoint": str(args.segmenter_checkpoint) if args.mode == "guided" else None,
            "best_macro_f1": best_f1,
            "seed": args.seed,
            "history": history,
        }
        torch.save(checkpoint, output / "last.pt")
        if is_best:
            torch.save(checkpoint, output / "best.pt")
        save_json(output / "history.json", history)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )
    save_json(output / "dataset.json", index.summary())


if __name__ == "__main__":
    main()
