"""Train the PetVision U-Net segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from petvision.config import IMAGE_SIZE, IGNORE_INDEX, RANDOM_SEED
from petvision.data import OxfordIIITPetIndex
from petvision.data.dataset import OxfordPetDataset
from petvision.data.splits import stratified_train_val
from petvision.data.transforms import PairedTransform
from petvision.models.unet import UNet
from petvision.training.loops import run_segmentation_epoch
from petvision.training.utils import save_json, set_seed


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)

def _load_resume(
    path: Path,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, list[dict[str, object]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    else:
        print("resume checkpoint has no optimizer state; restarting optimizer")
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_miou = float(checkpoint.get("best_miou", -1.0))
    history = list(checkpoint.get("history", []))
    return start_epoch, best_miou, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/segmentation"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
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

    set_seed(args.seed)
    device = _device(args.device)
    index = OxfordIIITPetIndex(args.root)
    train_records, val_records = stratified_train_val(
        index.records("trainval"), args.val_fraction, args.seed
    )
    if args.limit > 0:
        train_records = train_records[: args.limit]
        val_records = val_records[: max(1, args.limit // 4)]
    train_dataset = OxfordPetDataset(
        index,
        train_records,
        PairedTransform(args.image_size, train=True),
    )
    val_dataset = OxfordPetDataset(
        index,
        val_records,
        PairedTransform(args.image_size, train=False),
    )
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

    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_miou = -1.0
    history: list[dict[str, object]] = []
    if args.resume is not None:
        start_epoch, best_miou, history = _load_resume(
            args.resume,
            model,
            optimizer,
            device,
        )
        print(f"resuming from {args.resume} at epoch {start_epoch}")
    if start_epoch > args.epochs:
        raise SystemExit(
            f"checkpoint is already past requested epochs: "
            f"start={start_epoch} epochs={args.epochs}"
        )

    print(
        f"device={device} train={len(train_dataset)} "
        f"val={len(val_dataset)} start_epoch={start_epoch}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_segmentation_epoch(
            model, train_loader, device, criterion, optimizer
        )
        with torch.no_grad():
            val_metrics = run_segmentation_epoch(
                model, val_loader, device, criterion
            )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(row)
        is_best = val_metrics["miou"] > best_miou
        if is_best:
            best_miou = float(val_metrics["miou"])
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "image_size": args.image_size,
            "class_names": ["background", "foreground"],
            "best_miou": best_miou,
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
            f"val_miou={val_metrics['miou']:.4f} "
            f"val_fg_dice={val_metrics['foreground_dice']:.4f}"
        )
    save_json(output / "dataset.json", index.summary())


if __name__ == "__main__":
    main()
