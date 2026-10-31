from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from petvision.data import OxfordIIITPetIndex, trimap_to_binary, trimap_to_target


class OxfordPetIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "images").mkdir()
        (self.root / "annotations" / "trimaps").mkdir(parents=True)
        (self.root / "annotations" / "trainval.txt").write_text(
            "# Image CLASS-ID SPECIES-ID\n"
            "Egyptian_Mau_1 1 1\n"
            "Abyssinian_1 2 1\n",
            encoding="utf-8",
        )
        (self.root / "annotations" / "test.txt").write_text(
            "Abyssinian_2 2 1\n",
            encoding="utf-8",
        )
        for image_id in ("Egyptian_Mau_1", "Abyssinian_1", "Abyssinian_2"):
            Image.new("RGB", (4, 3), color=(10, 20, 30)).save(
                self.root / "images" / f"{image_id}.jpg"
            )
            Image.fromarray(
                np.array(
                    [
                        [1, 1, 3, 2],
                        [1, 3, 2, 2],
                        [3, 2, 2, 2],
                    ],
                    dtype=np.uint8,
                )
            ).save(self.root / "annotations" / "trimaps" / f"{image_id}.png")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_index_builds_class_mapping_and_splits(self) -> None:
        index = OxfordIIITPetIndex(self.root)
        self.assertEqual(index.num_classes, 2)
        self.assertEqual(index.class_names, ("Abyssinian", "Egyptian_Mau"))
        self.assertEqual(len(index.records("trainval")), 2)
        self.assertEqual(len(index.records("test")), 1)
        self.assertEqual(index.get("Abyssinian_2").split, "test")

    def test_trimap_conversion_preserves_ignored_border(self) -> None:
        raw = np.array([[1, 2, 3]], dtype=np.uint8)
        target = trimap_to_target(raw)
        display = trimap_to_binary(raw)
        np.testing.assert_array_equal(target, [[1, 0, 255]])
        np.testing.assert_array_equal(display, [[1, 0, 1]])


if __name__ == "__main__":
    unittest.main()
