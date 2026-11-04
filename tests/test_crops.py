from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from petvision.data import (
    OxfordIIITPetIndex,
    OxfordPetCropDataset,
    PairedTransform,
    crop_image_pair,
    load_crop_cache,
)


class CropDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "images").mkdir()
        (self.root / "annotations" / "trimaps").mkdir(parents=True)
        (self.root / "annotations" / "trainval.txt").write_text(
            "Abyssinian_1 1 1\n", encoding="utf-8"
        )
        (self.root / "annotations" / "test.txt").write_text(
            "Abyssinian_2 1 1\n", encoding="utf-8"
        )
        for image_id in ("Abyssinian_1", "Abyssinian_2"):
            Image.fromarray(
                np.array(
                    [
                        [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
                        [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
                    ],
                    dtype=np.uint8,
                )
            ).save(self.root / "images" / f"{image_id}.jpg")
            Image.fromarray(
                np.array([[2, 1, 1, 2], [2, 1, 1, 2]], dtype=np.uint8)
            ).save(self.root / "annotations" / "trimaps" / f"{image_id}.png")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_crop_pair_uses_same_box_for_image_and_trimap(self) -> None:
        image = Image.new("RGB", (10, 8), color=(1, 2, 3))
        trimap = Image.new("L", (10, 8), color=2)
        cropped_image, cropped_trimap = crop_image_pair(
            image, trimap, (0.2, 0.25, 0.8, 0.75)
        )
        self.assertEqual(cropped_image.size, (6, 4))
        self.assertEqual(cropped_trimap.size, cropped_image.size)

    def test_crop_dataset_requires_and_uses_cached_boxes(self) -> None:
        index = OxfordIIITPetIndex(self.root)
        cache_path = self.root / "crop-cache.json"
        cache_path.write_text(
            json.dumps(
                {
                    "metadata": {"source": "predicted"},
                    "boxes": {"Abyssinian_1": [0.25, 0.0, 0.75, 1.0]},
                }
            ),
            encoding="utf-8",
        )
        metadata, boxes = load_crop_cache(cache_path)
        self.assertEqual(metadata["source"], "predicted")
        dataset = OxfordPetCropDataset(
            index,
            index.records("trainval"),
            PairedTransform(image_size=4, train=False),
            boxes,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 4, 4))
        with self.assertRaises(ValueError):
            OxfordPetCropDataset(
                index,
                index.records("test"),
                PairedTransform(image_size=4, train=False),
                boxes,
            )


if __name__ == "__main__":
    unittest.main()
