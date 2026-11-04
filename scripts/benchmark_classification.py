"""Benchmark raw ResNet training batches without saving candidate weights."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from petvision.data import OxfordIIITPetIndex, OxfordPetDataset, PairedTransform
from petvision.data.splits import stratified_train_val
from petvision.models import build_classifier
from petvision.training.utils import dataloader_options, save_json, set_seed


def benchmark_case(args, index, records, batch_size: int, precision: str) -> dict:
    device = torch.device("cuda")
    set_seed(args.seed, deterministic=False)
    model = build_classifier(
        index.num_classes, args.architecture, pretrained=args.pretrained
    ).to(device).train()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        OxfordPetDataset(
            index, records, PairedTransform(args.image_size, train=True, color_jitter=True)
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_options(device, args.num_workers),
    )
    if len(loader) == 0:
        raise ValueError("The batch size exceeds the training set size")
    iterator = iter(loader)
    measured_loss = torch.zeros((), device=device)
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if step >= args.warmup:
            measured_loss += loss.detach()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    mean_loss = float((measured_loss / args.steps).item())
    if not torch.isfinite(measured_loss).item():
        raise FloatingPointError(f"Non-finite loss for batch={batch_size} {precision}")
    return {
        "batch_size": batch_size,
        "precision": precision,
        "status": "ok",
        "measured_steps": args.steps,
        "seconds": seconds,
        "samples_per_second": batch_size * args.steps / seconds,
        "mean_loss": mean_loss,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/classification_benchmark"))
    parser.add_argument("--architecture", choices=("resnet18", "resnet50"), default="resnet50")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--precisions", choices=("fp32", "bf16"), nargs="+", default=["fp32", "bf16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error("steps, warmup, and batch sizes must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA/ROCm GPU is required")
    index = OxfordIIITPetIndex(args.root)
    records, _ = stratified_train_val(index.records("trainval"), seed=args.split_seed)
    properties = torch.cuda.get_device_properties(0)
    result = {
        "architecture": args.architecture,
        "pretrained": args.pretrained,
        "image_size": args.image_size,
        "training_seed": args.seed,
        "split_seed": args.split_seed,
        "train_count": len(records),
        "num_workers": args.num_workers,
        "warmup_steps": args.warmup,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": properties.name,
        "total_memory_gib": properties.total_memory / 2**30,
        "measurement": "data loading, forward, backward, and AdamW; no validation, metrics, or checkpoint I/O",
        "cases": [],
    }
    for precision in args.precisions:
        for batch_size in args.batch_sizes:
            gc.collect()
            torch.cuda.empty_cache()
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                row = {"batch_size": batch_size, "precision": precision, "status": "unsupported"}
            else:
                try:
                    row = benchmark_case(args, index, records, batch_size, precision)
                except torch.cuda.OutOfMemoryError:
                    row = {"batch_size": batch_size, "precision": precision, "status": "out_of_memory"}
            result["cases"].append(row)
            save_json(args.output / "benchmark.json", result)
            print(row, flush=True)
    print(f"Benchmark saved to {args.output / 'benchmark.json'}; no weights were saved.")


if __name__ == "__main__":
    main()
