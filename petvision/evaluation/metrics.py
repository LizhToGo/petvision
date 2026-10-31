"""Streaming metrics for classification and binary segmentation."""

from __future__ import annotations

import numpy as np
import torch


class ClassificationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.correct = 0
        self.total = 0
        self.top5_correct = 0
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        predictions = logits.detach().argmax(dim=1).cpu().numpy()
        labels = target.detach().cpu().numpy()
        self.correct += int((predictions == labels).sum())
        self.total += int(labels.size)
        top_k = min(5, logits.shape[1])
        top_indices = logits.detach().topk(top_k, dim=1).indices.cpu().numpy()
        self.top5_correct += int(
            np.any(top_indices == labels[:, None], axis=1).sum()
        )
        for truth, prediction in zip(labels, predictions):
            self.confusion[int(truth), int(prediction)] += 1

    def compute(self) -> dict[str, object]:
        diagonal = np.diag(self.confusion).astype(np.float64)
        predicted = self.confusion.sum(axis=0).astype(np.float64)
        actual = self.confusion.sum(axis=1).astype(np.float64)
        precision = np.divide(
            diagonal,
            predicted,
            out=np.zeros_like(diagonal),
            where=predicted > 0,
        )
        recall = np.divide(
            diagonal,
            actual,
            out=np.zeros_like(diagonal),
            where=actual > 0,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(diagonal),
            where=(precision + recall) > 0,
        )
        return {
            "accuracy": self.correct / max(self.total, 1),
            "top5_accuracy": self.top5_correct / max(self.total, 1),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "confusion_matrix": self.confusion.copy(),
        }


class SegmentationMeter:
    def __init__(self, num_classes: int = 2) -> None:
        if num_classes != 2:
            raise ValueError("SegmentationMeter currently supports two classes")
        self.intersection = np.zeros(2, dtype=np.int64)
        self.union = np.zeros(2, dtype=np.int64)
        self.predicted = np.zeros(2, dtype=np.int64)
        self.actual = np.zeros(2, dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = logits.detach().argmax(dim=1)
        valid = target != 255
        for class_id in range(2):
            pred_class = (prediction == class_id) & valid
            true_class = (target == class_id) & valid
            intersection = (pred_class & true_class).sum().item()
            union = (pred_class | true_class).sum().item()
            self.intersection[class_id] += int(intersection)
            self.union[class_id] += int(union)
            self.predicted[class_id] += int(pred_class.sum().item())
            self.actual[class_id] += int(true_class.sum().item())

    def compute(self) -> dict[str, float]:
        iou = np.divide(
            self.intersection,
            self.union,
            out=np.ones(2, dtype=np.float64),
            where=self.union > 0,
        )
        dice_denominator = self.predicted + self.actual
        dice = np.divide(
            2 * self.intersection,
            dice_denominator,
            out=np.ones(2, dtype=np.float64),
            where=dice_denominator > 0,
        )
        return {
            "background_iou": float(iou[0]),
            "foreground_iou": float(iou[1]),
            "miou": float(iou.mean()),
            "background_dice": float(dice[0]),
            "foreground_dice": float(dice[1]),
        }
