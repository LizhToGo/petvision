"""End-to-end inference for segmentation, classification, and late fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from petvision.data.crops import box_from_mask, crop_image
from petvision.data.transforms import image_to_tensor
from petvision.models import UNet, classifier_from_checkpoint
from petvision.training.loops import apply_soft_mask


@dataclass
class Prediction:
    """Serializable pieces needed by the command-line and web UI."""

    mask: np.ndarray
    foreground_probability: np.ndarray
    overlay: Image.Image
    foreground_image: Image.Image
    raw_top5: list[tuple[str, float]]
    guided_top5: list[tuple[str, float]]
    fusion_top5: list[tuple[str, float]]


class PetVisionInference:
    """Load trained checkpoints once and perform real single-image inference."""

    def __init__(
        self,
        segmenter: UNet,
        raw_classifier: nn.Module,
        guided_classifier: nn.Module,
        class_names: tuple[str, ...],
        image_size: int,
        alpha: float,
        fusion_raw_weight: float,
        device: torch.device,
    ) -> None:
        self.segmenter = segmenter.eval()
        self.raw_classifier = raw_classifier.eval()
        self.guided_classifier = guided_classifier.eval()
        self.class_names = class_names
        self.image_size = image_size
        self.alpha = alpha
        if not 0.0 <= fusion_raw_weight <= 1.0:
            raise ValueError("fusion_raw_weight must be between 0 and 1")
        self.fusion_raw_weight = fusion_raw_weight
        self.device = device

    @classmethod
    def from_checkpoints(
        cls,
        segmentation_checkpoint: str | Path,
        raw_checkpoint: str | Path,
        guided_checkpoint: str | Path,
        device: str = "auto",
        fusion_raw_weight: float = 0.23,
    ) -> "PetVisionInference":
        if device == "auto":
            selected_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            selected_device = torch.device(device)
        segmentation = torch.load(
            segmentation_checkpoint,
            map_location=selected_device,
            weights_only=False,
        )
        raw = torch.load(
            raw_checkpoint,
            map_location=selected_device,
            weights_only=False,
        )
        guided = torch.load(
            guided_checkpoint,
            map_location=selected_device,
            weights_only=False,
        )
        class_names = tuple(raw["class_names"])
        if class_names != tuple(guided["class_names"]):
            raise ValueError("Raw and guided checkpoints have different classes")
        if raw["image_size"] != guided["image_size"]:
            raise ValueError("Raw and guided checkpoints use different image sizes")
        segmenter_model = UNet().to(selected_device)
        segmenter_model.load_state_dict(segmentation["model_state"])
        raw_model = classifier_from_checkpoint(raw, selected_device)
        guided_model = classifier_from_checkpoint(guided, selected_device)
        return cls(
            segmenter=segmenter_model,
            raw_classifier=raw_model,
            guided_classifier=guided_model,
            class_names=class_names,
            image_size=int(raw["image_size"]),
            alpha=float(guided.get("alpha", 0.25)),
            fusion_raw_weight=fusion_raw_weight,
            device=selected_device,
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 5) -> Prediction:
        image_tensor, resized = image_to_tensor(image, self.image_size)
        batch = image_tensor.unsqueeze(0).to(self.device)
        segmentation_logits = self.segmenter(batch)
        foreground_probability = segmentation_logits.softmax(dim=1)[0, 1]
        binary_mask = (foreground_probability >= 0.5).cpu().numpy().astype(np.uint8)
        guided_input = apply_soft_mask(batch, segmentation_logits, self.alpha)
        raw_probability = self.raw_classifier(batch).softmax(dim=1)[0]
        guided_probability = self.guided_classifier(guided_input).softmax(dim=1)[0]
        fusion_probability = (
            self.fusion_raw_weight * raw_probability
            + (1.0 - self.fusion_raw_weight) * guided_probability
        )
        raw_top5 = self._top_k_probabilities(raw_probability, top_k)
        guided_top5 = self._top_k_probabilities(guided_probability, top_k)
        fusion_top5 = self._top_k_probabilities(fusion_probability, top_k)

        rgb = np.asarray(resized, dtype=np.uint8)
        overlay = rgb.copy()
        overlay[binary_mask == 1] = (
            0.65 * overlay[binary_mask == 1] + np.array([0, 100, 0])
        ).astype(np.uint8)
        foreground = rgb * binary_mask[..., None]
        return Prediction(
            mask=binary_mask,
            foreground_probability=foreground_probability.cpu().numpy(),
            overlay=Image.fromarray(overlay),
            foreground_image=Image.fromarray(foreground),
            raw_top5=raw_top5,
            guided_top5=guided_top5,
            fusion_top5=fusion_top5,
        )

    def _top_k_probabilities(
        self,
        probabilities: torch.Tensor,
        top_k: int,
    ) -> list[tuple[str, float]]:
        values, indices = probabilities.topk(min(top_k, len(self.class_names)))
        return [
            (self.class_names[int(index)], float(value))
            for value, index in zip(values.cpu(), indices.cpu())
        ]


@dataclass
class CropPrediction:
    """Outputs from the segmentation-local-crop demo."""

    mask: np.ndarray
    foreground_probability: np.ndarray
    overlay: Image.Image
    crop_image: Image.Image
    crop_top5: list[tuple[str, float]]
    crop_box: tuple[float, float, float, float]
    crop_fallback: bool
    raw_top5: list[tuple[str, float]] | None


class PetVisionCropInference:
    """Run U-Net localization followed by a crop ResNet classifier."""

    def __init__(
        self,
        segmenter: UNet,
        crop_classifier: nn.Module,
        class_names: tuple[str, ...],
        image_size: int,
        threshold: float,
        margin: float,
        device: torch.device,
        segmentation_size: int,
        raw_classifier: nn.Module | None,
        raw_image_size: int,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        self.segmenter = segmenter.eval()
        self.crop_classifier = crop_classifier.eval()
        self.class_names = class_names
        self.image_size = image_size
        self.threshold = threshold
        self.margin = margin
        self.device = device
        self.segmentation_size = segmentation_size
        self.raw_classifier = raw_classifier.eval() if raw_classifier is not None else None
        self.raw_image_size = raw_image_size

    @classmethod
    def from_checkpoints(
        cls,
        segmentation_checkpoint: str | Path,
        crop_checkpoint: str | Path,
        device: str = "auto",
        raw_checkpoint: str | Path | None = None,
    ) -> "PetVisionCropInference":
        if device == "auto":
            selected_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            selected_device = torch.device(device)
        segmentation = torch.load(
            segmentation_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        crop = torch.load(
            crop_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if crop.get("mode") != "crop":
            raise ValueError("The classifier checkpoint is not a crop checkpoint")
        metadata = crop["crop_cache_metadata"]["path_metadata"]
        if metadata["source"] != "predicted":
            raise ValueError("Oracle crop checkpoints cannot be used for uploaded images")
        if int(segmentation.get("image_size", 256)) != int(metadata["image_size"]):
            raise ValueError("Segmenter resolution differs from the crop training cache")
        raw_model = None
        raw_image_size = int(crop["image_size"])
        if raw_checkpoint is not None:
            raw = torch.load(raw_checkpoint, map_location="cpu", weights_only=False)
            if raw.get("mode") != "raw":
                raise ValueError("The comparison checkpoint must be a raw classifier")
            if tuple(raw["class_names"]) != tuple(crop["class_names"]):
                raise ValueError("Raw and crop checkpoints have different classes")
            raw_model = classifier_from_checkpoint(raw, selected_device)
            raw_image_size = int(raw["image_size"])
        segmenter_model = UNet().to(selected_device)
        segmenter_model.load_state_dict(segmentation["model_state"])
        crop_model = classifier_from_checkpoint(crop, selected_device)
        return cls(
            segmenter=segmenter_model,
            crop_classifier=crop_model,
            class_names=tuple(crop["class_names"]),
            image_size=int(crop["image_size"]),
            threshold=float(metadata["threshold"]),
            margin=float(metadata["margin"]),
            device=selected_device,
            segmentation_size=int(metadata["image_size"]),
            raw_classifier=raw_model,
            raw_image_size=raw_image_size,
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 5) -> CropPrediction:
        image_rgb = image.convert("RGB")
        image_tensor, resized = image_to_tensor(image_rgb, self.segmentation_size)
        batch = image_tensor.unsqueeze(0).to(self.device)
        segmentation_logits = self.segmenter(batch)
        foreground_probability = segmentation_logits.softmax(dim=1)[0, 1]
        binary_mask = (
            foreground_probability >= self.threshold
        ).cpu().numpy().astype(np.uint8)
        crop_box, fallback = box_from_mask(binary_mask, self.margin)
        local_image = crop_image(image_rgb, crop_box)
        local_tensor, _ = image_to_tensor(local_image, self.image_size)
        local_logits = self.crop_classifier(
            local_tensor.unsqueeze(0).to(self.device)
        )
        local_probability = local_logits.softmax(dim=1)[0]
        raw_top5 = None
        if self.raw_classifier is not None:
            raw_tensor, _ = image_to_tensor(image_rgb, self.raw_image_size)
            raw_probability = self.raw_classifier(
                raw_tensor.unsqueeze(0).to(self.device)
            ).softmax(dim=1)[0]
            raw_top5 = self._top_k_probabilities(raw_probability, top_k)
        rgb = np.asarray(resized, dtype=np.uint8)
        overlay = rgb.copy()
        overlay[binary_mask == 1] = (
            0.65 * overlay[binary_mask == 1] + np.array([0, 100, 0])
        ).astype(np.uint8)
        return CropPrediction(
            mask=binary_mask,
            foreground_probability=foreground_probability.cpu().numpy(),
            overlay=Image.fromarray(overlay),
            crop_image=local_image,
            crop_top5=self._top_k_probabilities(local_probability, top_k),
            crop_box=crop_box,
            crop_fallback=fallback,
            raw_top5=raw_top5,
        )


    def _top_k_probabilities(
        self, probabilities: torch.Tensor, top_k: int
    ) -> list[tuple[str, float]]:
        values, indices = probabilities.topk(min(top_k, len(self.class_names)))
        return [
            (self.class_names[int(index)], float(value))
            for value, index in zip(values.cpu(), indices.cpu())
        ]
