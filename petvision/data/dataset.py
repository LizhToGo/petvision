"""PyTorch datasets for indexed Oxford-IIIT Pet records."""

from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import Dataset

from petvision.data.crops import CropBox, crop_image_pair
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


class OxfordPetCropDataset(OxfordPetDataset):
    """Return samples cropped by cached normalized foreground boxes."""

    def __init__(
        self,
        index: OxfordIIITPetIndex,
        records: Sequence[PetRecord],
        transform: PairedTransform,
        crop_boxes: Mapping[str, CropBox],
    ) -> None:
        super().__init__(index, records, transform)
        self.crop_boxes = dict(crop_boxes)
        missing = [
            record.image_id
            for record in self.records
            if record.image_id not in self.crop_boxes
        ]
        if missing:
            raise ValueError(
                f"Crop cache is missing {len(missing)} records, first={missing[0]!r}"
            )

    def __getitem__(self, item: int) -> dict[str, object]:
        record = self.records[item]
        image, trimap = self.index.load_images(record)
        image, trimap = crop_image_pair(
            image,
            trimap,
            self.crop_boxes[record.image_id],
        )
        image_tensor, target_tensor = self.transform(image, trimap)
        return {
            "image": image_tensor,
            "mask": target_tensor,
            "label": torch.tensor(record.label, dtype=torch.long),
            "image_id": record.image_id,
        }
