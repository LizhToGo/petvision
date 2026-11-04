"""Training and evaluation loops shared by command-line scripts."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from petvision.config import IGNORE_INDEX
from petvision.evaluation.metrics import ClassificationMeter, SegmentationMeter


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Foreground Dice loss, excluding the ambiguous trimap border."""

    probability = logits.softmax(dim=1)[:, 1]
    valid = target != IGNORE_INDEX
    foreground = target == 1
    probability = probability[valid]
    foreground = foreground[valid].float()
    if probability.numel() == 0:
        return logits.sum() * 0.0
    intersection = (probability * foreground).sum()
    denominator = probability.sum() + foreground.sum()
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice


def apply_soft_mask(
    images: torch.Tensor,
    segmentation_logits: torch.Tensor,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Reduce background strength while preserving a configurable context term."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    foreground_probability = segmentation_logits.softmax(dim=1)[:, 1:2]
    weight = alpha + (1.0 - alpha) * foreground_probability
    return images * weight

def apply_true_mask(
    images: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Apply the ground-truth mask for the oracle comparison experiment."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    keep = ((target == 1) | (target == IGNORE_INDEX)).unsqueeze(1)
    weight = alpha + (1.0 - alpha) * keep.float()
    return images * weight


def _guided_inputs(
    images: torch.Tensor,
    segmenter: nn.Module | None,
    alpha: float,
) -> torch.Tensor:
    if segmenter is None:
        return images
    with torch.no_grad():
        segmentation_logits = segmenter(images)
    return apply_soft_mask(images, segmentation_logits, alpha)


def run_classification_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    segmenter: nn.Module | None = None,
    alpha: float = 0.25,
    oracle_mask: bool = False,
    *,
    precision: str = "fp32",
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    if segmenter is not None:
        segmenter.eval()
    loss_sum = 0.0
    sample_count = 0
    meter = ClassificationMeter(model.classifier[-1].out_features)

    for batch in loader:
        non_blocking = device.type == "cuda"
        images = batch["image"].to(device, non_blocking=non_blocking)
        labels = batch["label"].to(device, non_blocking=non_blocking)
        if oracle_mask:
            images = apply_true_mask(
                images,
                batch["mask"].to(device, non_blocking=non_blocking),
                alpha,
            )
        else:
            images = _guided_inputs(images, segmenter, alpha)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
        ):
            logits = model(images)
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


def run_segmentation_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    dice_weight: float = 0.5,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    meter = SegmentationMeter()

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss = loss + dice_weight * dice_loss(logits, targets)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = images.shape[0]
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        meter.update(logits, targets)

    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(sample_count, 1)
    return metrics
