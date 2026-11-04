"""Predicted foreground crop cache and image cropping helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from PIL import Image

CropBox = tuple[float, float, float, float]


def load_crop_cache(path: str | Path) -> tuple[dict[str, object], dict[str, CropBox]]:
    """Load normalized crop boxes and metadata from a JSON cache."""

    cache_path = Path(path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    raw_boxes = payload.get("boxes")
    if not isinstance(metadata, dict) or not isinstance(raw_boxes, dict):
        raise ValueError(f"Invalid crop cache structure: {cache_path}")
    boxes: dict[str, CropBox] = {}
    for image_id, raw_box in raw_boxes.items():
        if not isinstance(image_id, str) or not isinstance(raw_box, list):
            raise ValueError(f"Invalid crop box for {image_id!r}")
        if len(raw_box) != 4:
            raise ValueError(f"Crop box must have four coordinates: {image_id!r}")
        box = tuple(float(value) for value in raw_box)
        if not all(0.0 <= value <= 1.0 for value in box):
            raise ValueError(f"Crop box is outside [0, 1]: {image_id!r}")
        left, top, right, bottom = box
        if not left < right or not top < bottom:
            raise ValueError(f"Crop box is empty: {image_id!r}")
        boxes[image_id] = box
    return metadata, boxes


def crop_image_pair(
    image: Image.Image,
    trimap: Image.Image,
    box: CropBox,
) -> tuple[Image.Image, Image.Image]:
    """Crop an image/trimap pair using a normalized box."""

    width, height = image.size
    left, top, right, bottom = box
    pixel_box = (
        max(0, min(width - 1, int(round(left * width)))),
        max(0, min(height - 1, int(round(top * height)))),
        max(1, min(width, int(round(right * width)))),
        max(1, min(height, int(round(bottom * height)))),
    )
    if pixel_box[2] <= pixel_box[0]:
        pixel_box = (pixel_box[0], pixel_box[1], min(width, pixel_box[0] + 1), pixel_box[3])
    if pixel_box[3] <= pixel_box[1]:
        pixel_box = (pixel_box[0], pixel_box[1], pixel_box[2], min(height, pixel_box[1] + 1))
    return image.crop(pixel_box), trimap.crop(pixel_box)


def crop_cache_summary(
    metadata: Mapping[str, object], boxes: Mapping[str, CropBox]
) -> dict[str, object]:
    """Return compact metadata suitable for experiment checkpoints."""

    return {
        "path_metadata": dict(metadata),
        "count": len(boxes),
    }
