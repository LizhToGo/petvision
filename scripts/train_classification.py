"""Train raw or segmentation-guided pet classification."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from petvision.config import IMAGE_SIZE, RANDOM_SEED
from petvision.data import (
    OxfordIIITPetIndex,
    OxfordPetCropDataset,
    OxfordPetDataset,
    PairedTransform,
    crop_cache_summary,
    load_crop_cache,
)
from petvision.data.splits import stratified_train_val
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
    if architecture not in ("resnet18", "resnet50"):
        return
    setter = getattr(model, "set_backbone_trainable", None)
    if setter is None:
        raise TypeError("ResNet model has no backbone trainability control")
    setter(trainable)


def _build_optimizer(
    model: nn.Module,
    architecture: str,
    schedule: str,
    learning_rate: float,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if architecture in ("resnet18", "resnet50") and schedule == "staged":
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
            ],
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_schedule: str,
    t_max: int,
    eta_min: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if lr_schedule == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, t_max),
        eta_min=eta_min,
    )




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mode", choices=("raw", "guided", "crop"), default="raw"
    )
    parser.add_argument(
        "--architecture",
        choices=("small_cnn", "resnet18", "resnet50"),
        default=None,
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="initialize ResNet-18 with ImageNet V1 or ResNet-50 with ImageNet V2",
    )
    parser.add_argument(
        "--augmentation",
        choices=("none", "color"),
        default=None,
        help="training image augmentation; color applies mild brightness/contrast/saturation jitter",
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        type=Path,
        default=Path("artifacts/segmentation/best.pt"),
    )
    parser.add_argument(
        "--crop-cache",
        type=Path,
        default=None,
        help="normalized foreground crop cache required by crop mode",
    )
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=None)
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
        "--lr-schedule",
        choices=("none", "cosine"),
        default=None,
        help="learning-rate schedule; cosine decays after each full-training epoch",
    )
    parser.add_argument(
        "--cosine-eta-min",
        type=float,
        default=None,
        help="minimum learning rate for cosine scheduling",
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
    parser.add_argument(
        "--precision", choices=("fp32", "bf16"), default=None,
        help="classifier training precision; validation always uses FP32",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="training randomness only; defaults to 42 or the resume checkpoint",
    )
    parser.add_argument(
        "--split-seed", type=int, default=None,
        help="fixed train/validation split; defaults to 42 or the resume checkpoint",
    )
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
    if args.mode == "crop" and (
        args.crop_cache is None or not args.crop_cache.is_file()
    ):
        raise SystemExit(
            "Crop mode requires an existing normalized crop cache: "
            f"{args.crop_cache}"
        )
    if args.head_epochs is not None and args.head_epochs < 0:
        raise SystemExit("--head-epochs must be non-negative")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 0:
        raise SystemExit("--early-stopping-patience must be non-negative")
    device = _device(args.device)
    resume_checkpoint = (
        torch.load(args.resume, map_location=device, weights_only=False)
        if args.resume is not None
        else None
    )
    saved = resume_checkpoint or {}
    checkpoint_precision = (
        saved.get("precision", "fp32") if resume_checkpoint is not None else None
    )
    if args.precision is not None and checkpoint_precision is not None:
        if args.precision != checkpoint_precision:
            raise SystemExit("--precision does not match the resume checkpoint")
    precision = args.precision or checkpoint_precision or "fp32"
    if precision == "bf16":
        if device.type != "cuda":
            raise SystemExit("BF16 training requires a CUDA/ROCm GPU")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise SystemExit("The selected CUDA/ROCm GPU does not support BF16")
    if args.seed is None:
        args.seed = int(saved.get("training_seed", RANDOM_SEED))
    if args.split_seed is None:
        args.split_seed = int(saved.get("split_seed", RANDOM_SEED))
    if args.val_fraction is None:
        args.val_fraction = float(saved.get("val_fraction", 0.15))
    weight_decay = (
        args.weight_decay if args.weight_decay is not None
        else float(saved.get("weight_decay", 0.01))
    )
    set_seed(args.seed, deterministic=not args.fast)
    checkpoint_architecture = (
        resume_checkpoint.get("architecture", "small_cnn")
        if resume_checkpoint is not None
        else None
    )
    if args.architecture is not None and checkpoint_architecture is not None:
        if args.architecture != checkpoint_architecture:
            raise SystemExit("--architecture does not match the resume checkpoint")
    architecture = args.architecture or checkpoint_architecture or "small_cnn"
    if args.pretrained and architecture not in ("resnet18", "resnet50"):
        raise SystemExit("--pretrained requires resnet18 or resnet50")
    checkpoint_schedule = (
        resume_checkpoint.get("schedule", "flat")
        if resume_checkpoint is not None
        else None
    )
    if args.schedule is not None and checkpoint_schedule is not None:
        if args.schedule != checkpoint_schedule:
            raise SystemExit("--schedule does not match the resume checkpoint")
    schedule = args.schedule or checkpoint_schedule or "flat"
    checkpoint_lr_schedule = (
        resume_checkpoint.get("lr_schedule", "none")
        if resume_checkpoint is not None
        else None
    )
    if args.lr_schedule is not None and checkpoint_lr_schedule is not None:
        if args.lr_schedule != checkpoint_lr_schedule:
            raise SystemExit("--lr-schedule does not match the resume checkpoint")
    lr_schedule = args.lr_schedule or checkpoint_lr_schedule or "none"
    checkpoint_augmentation = (
        resume_checkpoint.get("augmentation", "none")
        if resume_checkpoint is not None
        else None
    )
    if args.augmentation is not None and checkpoint_augmentation is not None:
        if args.augmentation != checkpoint_augmentation:
            raise SystemExit("--augmentation does not match the resume checkpoint")
    augmentation = args.augmentation or checkpoint_augmentation or "none"
    if schedule == "staged" and architecture not in ("resnet18", "resnet50"):
        raise SystemExit("--schedule staged requires resnet18 or resnet50")
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
    checkpoint_cosine_eta_min = (
        resume_checkpoint.get("cosine_eta_min")
        if resume_checkpoint is not None
        else None
    )
    cosine_eta_min = (
        args.cosine_eta_min
        if args.cosine_eta_min is not None
        else float(checkpoint_cosine_eta_min or 0.0)
    )
    scheduler_t_max = (
        args.epochs - head_epochs if schedule == "staged" else args.epochs
    )
    saved_scheduler_t_max = saved.get("scheduler_t_max")
    if (
        resume_checkpoint is not None
        and lr_schedule == "cosine"
        and saved_scheduler_t_max is not None
        and scheduler_t_max != int(saved_scheduler_t_max)
    ):
        raise SystemExit(
            "--epochs/head-epochs do not match the cosine resume checkpoint"
        )
    if lr_schedule == "cosine":
        minimum_initial_lr = (
            min(backbone_learning_rate, head_learning_rate)
            if schedule == "staged"
            else learning_rate
        )
        if not 0.0 <= cosine_eta_min < minimum_initial_lr:
            raise SystemExit(
                "--cosine-eta-min must be non-negative and below the initial learning rates"
            )
    else:
        cosine_eta_min = 0.0
        scheduler_t_max = 0

    index = OxfordIIITPetIndex(args.root)
    crop_metadata: dict[str, object] | None = None
    crop_boxes = None
    if args.mode == "crop":
        crop_metadata, crop_boxes = load_crop_cache(args.crop_cache)
    train_records, val_records = stratified_train_val(
        index.records("trainval"), args.val_fraction, args.split_seed
    )
    if args.limit > 0:
        train_records = train_records[: args.limit]
        val_records = val_records[: max(1, args.limit // 4)]
    split = {
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "train_image_ids": [record.image_id for record in train_records],
        "validation_image_ids": [record.image_id for record in val_records],
    }
    if "split" in saved and split != saved["split"]:
        raise SystemExit("Training/validation records differ from the resume checkpoint")
    transform_train = PairedTransform(
        args.image_size,
        train=True,
        color_jitter=augmentation == "color",
    )
    transform_val = PairedTransform(args.image_size, train=False)
    if args.mode == "crop":
        train_dataset = OxfordPetCropDataset(
            index, train_records, transform_train, crop_boxes
        )
        val_dataset = OxfordPetCropDataset(
            index, val_records, transform_val, crop_boxes
        )
    else:
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
    if architecture in ("resnet18", "resnet50"):
        # Segmenter construction must not change the classifier's initialization.
        set_seed(args.seed, deterministic=not args.fast)
    model = build_classifier(
        num_classes=index.num_classes,
        architecture=architecture,
        pretrained=args.pretrained and resume_checkpoint is None,
    ).to(device)
    pretrained_source = (
        saved.get("pretrained_source") if resume_checkpoint is not None
        else getattr(model, "pretrained_source", None)
    )
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
        weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        lr_schedule,
        scheduler_t_max,
        cosine_eta_min,
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
        if scheduler is not None and resume_checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        print(f"resuming from {args.resume} at epoch {start_epoch}")
    if start_epoch > args.epochs:
        raise SystemExit(
            f"checkpoint is already past requested epochs: "
            f"start={start_epoch} epochs={args.epochs}"
        )

    config = {
        "root": str(args.root),
        "image_size": args.image_size,
        "class_names": list(index.class_names),
        "mode": args.mode,
        "architecture": architecture,
        "pretrained_source": pretrained_source,
        "augmentation": augmentation,
        "alpha": args.alpha,
        "learning_rate": learning_rate,
        "backbone_learning_rate": backbone_learning_rate,
        "head_learning_rate": head_learning_rate,
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "schedule": schedule,
        "lr_schedule": lr_schedule,
        "cosine_eta_min": cosine_eta_min,
        "scheduler_t_max": scheduler_t_max,
        "head_epochs": head_epochs,
        "training_seed": args.seed,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "precision": precision,
        "validation_precision": "fp32",
        "segmenter_checkpoint": (
            str(args.segmenter_checkpoint) if args.mode == "guided" else None
        ),
        "crop_cache": str(args.crop_cache) if args.mode == "crop" else None,
        "crop_cache_metadata": (
            crop_cache_summary(crop_metadata, crop_boxes)
            if args.mode == "crop"
            else None
        ),
        "fast": args.fast,
    }
    save_json(output / "config.json", config)
    save_json(output / "split.json", split)
    save_json(output / "dataset.json", index.summary())

    print(
        f"device={device} mode={args.mode} architecture={architecture} "
        f"schedule={schedule} augmentation={augmentation} fast={args.fast} "
        f"train={len(train_dataset)} val={len(val_dataset)} "
        f"split_seed={args.split_seed} training_seed={args.seed} "
        f"precision={precision} validation_precision=fp32 "
        f"lr_schedule={lr_schedule} start_epoch={start_epoch}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        backbone_trainable = schedule != "staged" or epoch > head_epochs
        _set_backbone_trainable(model, architecture, backbone_trainable)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
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
            precision=precision,
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
        current_learning_rates = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
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
            "learning_rates": current_learning_rates,
        }
        history.append(row)
        if device.type == "cuda":
            row["memory"] = {
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            }
        if scheduler is not None and (
            schedule != "staged" or stage == "full"
        ):
            scheduler.step()
        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = {
            **config,
            "split": split,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "epoch": epoch,
            "best_macro_f1": best_f1,
            "epochs_without_improvement": epochs_without_improvement,
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
            f"lr={','.join(f'{value:.2e}' for value in current_learning_rates)} "
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


if __name__ == "__main__":
    main()
