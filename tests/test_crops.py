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
from petvision.data.crops import box_from_mask


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

    def test_crop_pair_preserves_original_pixels_and_mask_alignment(self) -> None:
        pixels = np.arange(80, dtype=np.uint8).reshape(8, 10)
        image = Image.fromarray(np.stack([pixels, pixels, pixels], axis=-1))
        trimap = Image.fromarray(pixels)
        cropped_image, cropped_trimap = crop_image_pair(
            image, trimap, (0.2, 0.25, 0.8, 0.75)
        )
        np.testing.assert_array_equal(np.asarray(cropped_image)[:, :, 0], pixels[2:6, 2:8])
        np.testing.assert_array_equal(np.asarray(cropped_trimap), pixels[2:6, 2:8])

    def test_foreground_box_uses_exclusive_edges_and_relative_margin(self) -> None:
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[10:50, 60:140] = 1
        box, fallback = box_from_mask(mask, margin=0.1)
        np.testing.assert_allclose(box, (0.26, 0.06, 0.74, 0.54))
        self.assertFalse(fallback)

    def test_empty_foreground_falls_back_but_full_foreground_does_not(self) -> None:
        empty_box, empty_fallback = box_from_mask(np.zeros((8, 10)), margin=0.15)
        full_box, full_fallback = box_from_mask(np.ones((8, 10)), margin=0.15)
        self.assertEqual(empty_box, (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(full_box, empty_box)
        self.assertTrue(empty_fallback)
        self.assertFalse(full_fallback)

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
        np.testing.assert_array_equal(sample["mask"].numpy(), np.ones((4, 4), dtype=np.int64))
        with self.assertRaises(ValueError):
            OxfordPetCropDataset(
                index,
                index.records("test"),
                PairedTransform(image_size=4, train=False),
                boxes,
            )


if __name__ == "__main__":
    unittest.main()
