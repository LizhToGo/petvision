"""PyTorch datasets for indexed Oxford-IIIT Pet records."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from petvision.data.oxford_pet import OxfordIIITPetIndex, PetRecord
from petvision.data.transforms import PairedTransform


class OxfordPetDataset(Dataset):
    """Return normalized images, segmentation targets, labels, and ids."""

    def __init__(
        self,
        index: OxfordIIITPetIndex,
        records: Sequence[PetRecord],
        transform: PairedTransform,
    ) -> None:
        self.index = index
        self.records = tuple(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, object]:
        record = self.records[item]
        image, trimap = self.index.load_images(record)
        image_tensor, target_tensor = self.transform(image, trimap)
        return {
            "image": image_tensor,
            "mask": target_tensor,
            "label": torch.tensor(record.label, dtype=torch.long),
            "image_id": record.image_id,
        }
