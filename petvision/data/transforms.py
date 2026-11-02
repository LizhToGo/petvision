"""Paired image/trimap transforms without torchvision."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

from petvision.config import IMAGE_SIZE
from petvision.data.oxford_pet import trimap_to_target


def image_to_tensor(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> tuple[torch.Tensor, Image.Image]:
    """Resize an RGB image and return its normalized tensor plus preview image."""

    resized = image.convert("RGB").resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    image_array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype).view(3, 1, 1)
    return (image_tensor - mean_tensor) / std_tensor, resized


@dataclass(frozen=True)
class PairedTransform:
    """Resize, optionally augment, and convert an image/trimap pair."""

    image_size: int = IMAGE_SIZE
    train: bool = False
    color_jitter: bool = False
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __call__(
        self,
        image: Image.Image,
        trimap: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        trimap = trimap.resize(size, Image.Resampling.NEAREST)
        if self.train and self.color_jitter:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.8, 1.2)
            )
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.8, 1.2)
            )
            image = ImageEnhance.Color(image).enhance(
                random.uniform(0.8, 1.2)
            )
        if self.train and random.random() < 0.5:
            image = ImageOps.mirror(image)
            trimap = ImageOps.mirror(trimap)
        image_tensor, _ = image_to_tensor(
            image,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )

        target_array = trimap_to_target(trimap)
        target_tensor = torch.from_numpy(target_array.astype(np.int64))
        return image_tensor, target_tensor
