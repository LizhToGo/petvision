"""End-to-end inference for segmentation and two classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from petvision.data.transforms import image_to_tensor
from petvision.models import SmallCNN, UNet
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


class PetVisionInference:
    """Load trained checkpoints once and perform real single-image inference."""

    def __init__(
        self,
        segmenter: UNet,
        raw_classifier: SmallCNN,
        guided_classifier: SmallCNN,
        class_names: tuple[str, ...],
        image_size: int,
        alpha: float,
        device: torch.device,
    ) -> None:
        self.segmenter = segmenter.eval()
        self.raw_classifier = raw_classifier.eval()
        self.guided_classifier = guided_classifier.eval()
        self.class_names = class_names
        self.image_size = image_size
        self.alpha = alpha
        self.device = device

    @classmethod
    def from_checkpoints(
        cls,
        segmentation_checkpoint: str | Path,
        raw_checkpoint: str | Path,
        guided_checkpoint: str | Path,
        device: str = "auto",
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
        raw_model = SmallCNN(num_classes=len(class_names)).to(selected_device)
        raw_model.load_state_dict(raw["model_state"])
        guided_model = SmallCNN(num_classes=len(class_names)).to(selected_device)
        guided_model.load_state_dict(guided["model_state"])
        return cls(
            segmenter=segmenter_model,
            raw_classifier=raw_model,
            guided_classifier=guided_model,
            class_names=class_names,
            image_size=int(raw["image_size"]),
            alpha=float(guided.get("alpha", 0.25)),
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
        raw_logits = self.raw_classifier(batch)
        guided_logits = self.guided_classifier(guided_input)
        raw_top5 = self._top_k(raw_logits, top_k)
        guided_top5 = self._top_k(guided_logits, top_k)

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
        )

    def _top_k(
        self,
        logits: torch.Tensor,
        top_k: int,
    ) -> list[tuple[str, float]]:
        probabilities = logits.softmax(dim=1)[0]
        values, indices = probabilities.topk(min(top_k, len(self.class_names)))
        return [
            (self.class_names[int(index)], float(value))
            for value, index in zip(values.cpu(), indices.cpu())
        ]
