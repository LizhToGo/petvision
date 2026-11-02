"""Train raw or segmentation-guided pet classification."""

from __future__ import annotations

import argparse
import time
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
from petvision.training.utils import dataloader_options, save_json, set_seed


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_resume(
    checkpoint: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, list[dict[str, object]], int]:
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    else:
        print("resume checkpoint has no optimizer state; restarting optimizer")
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_f1 = float(checkpoint.get("best_macro_f1", -1.0))
    history = list(checkpoint.get("history", []))
    epochs_without_improvement = int(
        checkpoint.get("epochs_without_improvement", 0)
    )
    return start_epoch, best_f1, history, epochs_without_improvement


def _load_segmenter(path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _set_backbone_trainable(
    model: nn.Module,
    architecture: str,
    trainable: bool,
) -> None:
    if architecture != "resnet18":
        return
    setter = getattr(model, "set_backbone_trainable", None)
    if setter is None:
        raise TypeError("ResNet-18 model has no backbone trainability control")
    setter(trainable)


def _build_optimizer(
    model: nn.Module,
    architecture: str,
    schedule: str,
    learning_rate: float,
    backbone_learning_rate: float,
    head_learning_rate: float,
) -> torch.optim.Optimizer:
    if architecture == "resnet18" and schedule == "staged":
        return torch.optim.AdamW(
            [
                {
                    "params": getattr(model, "backbone").parameters(),
                    "lr": backbone_learning_rate,
                },
                {
                    "params": getattr(model, "classifier").parameters(),
                    "lr": head_learning_rate,
                },
            ]
        )
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("raw", "guided"), default="raw")
    parser.add_argument(
        "--architecture",
        choices=("small_cnn", "resnet18"),
        default=None,
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="initialize ResNet-18 with ImageNet-1K V1 weights",
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
    parser.add_argument(
        "--schedule",
        choices=("flat", "staged"),
        default=None,
        help="staged freezes the ResNet backbone before fine-tuning it",
    )
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=None,
        help="head-only epochs for the staged ResNet schedule",
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help="learning rate for the ResNet backbone in staged mode",
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=None,
        help="learning rate for the classifier head in staged mode",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="stop after this many non-improving validation epochs; 0 disables it",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="allow nondeterministic GPU algorithms and benchmark fixed-size kernels",
    )
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
    if args.head_epochs is not None and args.head_epochs < 0:
        raise SystemExit("--head-epochs must be non-negative")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 0:
        raise SystemExit("--early-stopping-patience must be non-negative")
    set_seed(args.seed, deterministic=not args.fast)
    device = _device(args.device)
    resume_checkpoint = (
        torch.load(args.resume, map_location=device, weights_only=False)
        if args.resume is not None
        else None
    )
    checkpoint_architecture = (
        resume_checkpoint.get("architecture", "small_cnn")
        if resume_checkpoint is not None
        else None
    )
    if args.architecture is not None and checkpoint_architecture is not None:
        if args.architecture != checkpoint_architecture:
            raise SystemExit("--architecture does not match the resume checkpoint")
    architecture = args.architecture or checkpoint_architecture or "small_cnn"
    if args.pretrained and architecture != "resnet18":
        raise SystemExit("--pretrained is only supported for resnet18")
    checkpoint_schedule = (
        resume_checkpoint.get("schedule", "flat")
        if resume_checkpoint is not None
        else None
    )
    if args.schedule is not None and checkpoint_schedule is not None:
        if args.schedule != checkpoint_schedule:
            raise SystemExit("--schedule does not match the resume checkpoint")
    schedule = args.schedule or checkpoint_schedule or "flat"
    if schedule == "staged" and architecture != "resnet18":
        raise SystemExit("--schedule staged is only supported for resnet18")
    pretrained_source = (
        resume_checkpoint.get("pretrained_source")
        if resume_checkpoint is not None
        else (
            "torchvision:ResNet18_Weights.IMAGENET1K_V1"
            if args.pretrained
            else None
        )
    )
    head_epochs = args.head_epochs
    if head_epochs is None:
        head_epochs = int(
            resume_checkpoint.get("head_epochs", 2)
            if resume_checkpoint is not None
            else 2
        )
    if schedule == "staged" and head_epochs >= args.epochs:
        raise SystemExit("--head-epochs must be smaller than --epochs")
    learning_rate = args.learning_rate
    backbone_learning_rate = args.backbone_learning_rate
    if backbone_learning_rate is None:
        backbone_learning_rate = float(
            resume_checkpoint.get("backbone_learning_rate", learning_rate / 10.0)
            if resume_checkpoint is not None
            else learning_rate / 10.0
        )
    head_learning_rate = args.head_learning_rate
    if head_learning_rate is None:
        head_learning_rate = float(
            resume_checkpoint.get("head_learning_rate", learning_rate)
            if resume_checkpoint is not None
            else learning_rate
        )
    early_stopping_patience = args.early_stopping_patience
    if early_stopping_patience is None:
        early_stopping_patience = int(
            resume_checkpoint.get("early_stopping_patience", 0)
            if resume_checkpoint is not None
            else 0
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
        **dataloader_options(device, args.num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_options(device, args.num_workers),
    )

    segmenter = None
    if args.mode == "guided":
        segmenter = _load_segmenter(args.segmenter_checkpoint, device)
    if architecture == "resnet18":
        # Segmenter construction must not change the classifier's initialization.
        set_seed(args.seed, deterministic=not args.fast)
    model = build_classifier(
        num_classes=index.num_classes,
        architecture=architecture,
        pretrained=args.pretrained and resume_checkpoint is None,
    ).to(device)
    start_epoch_hint = (
        int(resume_checkpoint.get("epoch", 0)) + 1
        if resume_checkpoint is not None
        else 1
    )
    backbone_trainable = schedule != "staged" or start_epoch_hint > head_epochs
    _set_backbone_trainable(model, architecture, backbone_trainable)
    optimizer = _build_optimizer(
        model,
        architecture,
        schedule,
        learning_rate,
        backbone_learning_rate,
        head_learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    output = args.output or (
        Path("artifacts/classification/raw")
        if architecture == "small_cnn"
        else Path("artifacts/classification") / architecture / args.mode
    )
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_f1 = -1.0
    history: list[dict[str, object]] = []
    epochs_without_improvement = 0
    if resume_checkpoint is not None:
        start_epoch, best_f1, history, epochs_without_improvement = _load_resume(
            resume_checkpoint,
            model,
            optimizer,
        )
        print(f"resuming from {args.resume} at epoch {start_epoch}")
    if start_epoch > args.epochs:
        raise SystemExit(
            f"checkpoint is already past requested epochs: "
            f"start={start_epoch} epochs={args.epochs}"
        )

    print(
        f"device={device} mode={args.mode} architecture={architecture} "
        f"schedule={schedule} fast={args.fast} train={len(train_dataset)} "
        f"val={len(val_dataset)} start_epoch={start_epoch}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        backbone_trainable = schedule != "staged" or epoch > head_epochs
        _set_backbone_trainable(model, architecture, backbone_trainable)
        stage = "full" if backbone_trainable else "head"
        train_started = time.perf_counter()
        train_metrics = run_classification_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            segmenter,
            args.alpha,
        )
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
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
        validation_seconds = time.perf_counter() - validation_started
        row = {
            "epoch": epoch,
            "stage": stage,
            "timing": {
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "train_samples_per_second": len(train_dataset)
                / max(train_seconds, 1e-9),
            },
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(row)
        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
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
            "learning_rate": learning_rate,
            "backbone_learning_rate": backbone_learning_rate,
            "head_learning_rate": head_learning_rate,
            "schedule": schedule,
            "head_epochs": head_epochs,
            "early_stopping_patience": early_stopping_patience,
            "val_fraction": args.val_fraction,
            "batch_size": args.batch_size,
            "segmenter_checkpoint": (
                str(args.segmenter_checkpoint) if args.mode == "guided" else None
            ),
            "best_macro_f1": best_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "fast": args.fast,
            "history": history,
        }
        torch.save(checkpoint, output / "last.pt")
        if is_best:
            torch.save(checkpoint, output / "best.pt")
        save_json(output / "history.json", history)
        print(
            f"epoch={epoch:03d} stage={stage} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"train_s={train_seconds:.1f} "
            f"val_s={validation_seconds:.1f} "
            f"samples_s={len(train_dataset) / max(train_seconds, 1e-9):.1f}"
        )
        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"early stopping at epoch={epoch}; "
                f"patience={early_stopping_patience}"
            )
            break
    save_json(output / "dataset.json", index.summary())


if __name__ == "__main__":
    main()
