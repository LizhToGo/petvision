"""Tune and evaluate raw/guided probability late fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from petvision.data import OxfordIIITPetIndex
from petvision.data.dataset import OxfordPetDataset
from petvision.data.splits import stratified_train_val
from petvision.data.transforms import PairedTransform
from petvision.evaluation import ClassificationMeter
from petvision.models import UNet, classifier_from_checkpoint
from petvision.training import apply_soft_mask
from petvision.training.utils import dataloader_options, save_json, set_seed


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_models(
    segmentation_path: Path,
    raw_path: Path,
    guided_path: Path,
    device: torch.device,
) -> tuple[UNet, nn.Module, nn.Module, int, float, tuple[str, ...]]:
    segmentation_state = torch.load(
        segmentation_path,
        map_location=device,
        weights_only=False,
    )
    raw_state = torch.load(raw_path, map_location=device, weights_only=False)
    guided_state = torch.load(
        guided_path,
        map_location=device,
        weights_only=False,
    )
    class_names = tuple(raw_state["class_names"])
    if class_names != tuple(guided_state["class_names"]):
        raise ValueError("Raw and guided checkpoints use different classes")
    segmenter = UNet().to(device)
    segmenter.load_state_dict(segmentation_state["model_state"])
    raw_model = classifier_from_checkpoint(raw_state, device)
    guided_model = classifier_from_checkpoint(guided_state, device)
    return (
        segmenter.eval(),
        raw_model.eval(),
        guided_model.eval(),
        int(raw_state["image_size"]),
        float(guided_state.get("alpha", 0.25)),
        class_names,
    )


def _collect_probabilities(
    loader: DataLoader,
    segmenter: UNet,
    raw_model: nn.Module,
    guided_model: nn.Module,
    device: torch.device,
    alpha: float,
    tta_flip: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_batches: list[np.ndarray] = []
    guided_batches: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=device.type == "cuda",
            )

            def predict(
                inputs: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                raw_probability = raw_model(inputs).softmax(dim=1)
                segmentation_logits = segmenter(inputs)
                guided_images = apply_soft_mask(
                    inputs,
                    segmentation_logits,
                    alpha,
                )
                guided_probability = guided_model(guided_images).softmax(dim=1)
                return raw_probability, guided_probability

            raw_probability, guided_probability = predict(images)
            if tta_flip:
                flipped = torch.flip(images, dims=(-1,))
                flipped_raw, flipped_guided = predict(flipped)
                raw_probability = 0.5 * (raw_probability + flipped_raw)
                guided_probability = 0.5 * (
                    guided_probability + flipped_guided
                )
            raw_batches.append(raw_probability.cpu().numpy())
            guided_batches.append(guided_probability.cpu().numpy())
            labels.append(batch["label"].numpy())
    return (
        np.concatenate(raw_batches),
        np.concatenate(guided_batches),
        np.concatenate(labels),
    )


def _metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    logits = torch.from_numpy(np.log(np.clip(probability, 1e-8, 1.0)))
    target = torch.from_numpy(labels.astype(np.int64))
    meter = ClassificationMeter(probability.shape[1])
    meter.update(logits, target)
    return meter.compute()


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/fusion"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tta-flip",
        action="store_true",
        help="average predictions from the original and horizontally flipped image",
    )
    args = parser.parse_args()

    if not 0.0 < args.lambda_step <= 1.0:
        raise SystemExit("--lambda-step must be in (0, 1]")
    set_seed(args.seed)
    device = _device(args.device)
    index = OxfordIIITPetIndex(args.root)
    segmenter, raw_model, guided_model, image_size, alpha, class_names = (
        _load_models(
            args.segmentation_checkpoint,
            args.raw_checkpoint,
            args.guided_checkpoint,
            device,
        )
    )
    train_records, val_records = stratified_train_val(
        index.records("trainval"), seed=args.seed
    )
    test_records = index.records("test")
    if args.limit > 0:
        val_records = val_records[: args.limit]
        test_records = test_records[: args.limit]
    transform = PairedTransform(image_size, train=False)
    val_loader = DataLoader(
        OxfordPetDataset(index, val_records, transform),
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_options(device, args.num_workers),
    )
    test_loader = DataLoader(
        OxfordPetDataset(index, test_records, transform),
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_options(device, args.num_workers),
    )

    val_raw, val_guided, val_labels = _collect_probabilities(
        val_loader,
        segmenter,
        raw_model,
        guided_model,
        device,
        alpha,
        args.tta_flip,
    )
    test_raw, test_guided, test_labels = _collect_probabilities(
        test_loader,
        segmenter,
        raw_model,
        guided_model,
        device,
        alpha,
        args.tta_flip,
    )

    lambdas = np.arange(
        0.0,
        1.0 + args.lambda_step * 0.5,
        args.lambda_step,
    )
    search: list[dict[str, object]] = []
    best_lambda = 0.0
    best_key = (-1.0, -1.0, -1.0)
    for raw_weight in lambdas:
        probability = raw_weight * val_raw + (1.0 - raw_weight) * val_guided
        result = _metrics(probability, val_labels)
        key = (
            float(result["macro_f1"]),
            float(result["accuracy"]),
            -abs(float(raw_weight) - 0.5),
        )
        search.append(
            {
                "raw_weight": float(raw_weight),
                "guided_weight": float(1.0 - raw_weight),
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "top5_accuracy": result["top5_accuracy"],
            }
        )
        if key > best_key:
            best_key = key
            best_lambda = float(raw_weight)

    def evaluate_split(
        raw_probability: np.ndarray,
        guided_probability: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, object]:
        fusion_probability = (
            best_lambda * raw_probability
            + (1.0 - best_lambda) * guided_probability
        )
        return {
            "raw": _metrics(raw_probability, labels),
            "guided": _metrics(guided_probability, labels),
            "fusion": _metrics(fusion_probability, labels),
        }

    result = {
        "dataset": index.summary(),
        "class_names": list(class_names),
        "metadata": {
            "selection_metric": "validation_macro_f1",
            "tta_flip": args.tta_flip,
            "raw_weight": best_lambda,
            "guided_weight": 1.0 - best_lambda,
        },
        "validation": evaluate_split(val_raw, val_guided, val_labels),
        "test": evaluate_split(test_raw, test_guided, test_labels),
        "search": search,
    }
    save_json(args.output / "metrics.json", result)
    print(
        "selected raw_weight=%.2f guided_weight=%.2f "
        "val_f1=%.4f test_f1=%.4f test_acc=%.4f"
        % (
            best_lambda,
            1.0 - best_lambda,
            result["validation"]["fusion"]["macro_f1"],
            result["test"]["fusion"]["macro_f1"],
            result["test"]["fusion"]["accuracy"],
        )
    )


if __name__ == "__main__":
    main()
