"""Compare raw and guided classifiers under controlled background changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from petvision.data import OxfordIIITPetIndex, trimap_to_binary, trimap_to_target
from petvision.data.oxford_pet import PetRecord
from petvision.data.transforms import image_to_tensor
from petvision.models import SmallCNN, UNet
from petvision.training import run_classification_epoch
from petvision.training.utils import save_json, set_seed


class BackgroundDataset(Dataset):
    def __init__(
        self,
        index: OxfordIIITPetIndex,
        records: tuple[PetRecord, ...],
        image_size: int,
        mode: str,
    ) -> None:
        self.index = index
        self.records = records
        self.image_size = image_size
        self.mode = mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, object]:
        record = self.records[item]
        image, trimap = self.index.load_images(record)
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        trimap = trimap.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )
        mask = trimap_to_binary(trimap)
        if self.mode == "blur":
            background = image.filter(ImageFilter.GaussianBlur(radius=16))
        elif self.mode == "gray":
            background = Image.new("RGB", image.size, (128, 128, 128))
        elif self.mode == "original":
            background = image
        else:
            raise ValueError(f"Unknown background mode: {self.mode}")
        mask_image = Image.fromarray(mask * 255, mode="L")
        composite = Image.composite(image, background, mask_image)
        image_tensor, _ = image_to_tensor(composite, self.image_size)
        target = torch.from_numpy(trimap_to_target(trimap).astype("int64"))
        return {
            "image": image_tensor,
            "mask": target,
            "label": torch.tensor(record.label, dtype=torch.long),
            "image_id": record.image_id,
        }


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load(
    segmentation_path: Path,
    raw_path: Path,
    guided_path: Path,
    device: torch.device,
) -> tuple[UNet, SmallCNN, SmallCNN, int, float]:
    segmentation_state = torch.load(
        segmentation_path,
        map_location=device,
        weights_only=False,
    )
    raw_state = torch.load(raw_path, map_location=device, weights_only=False)
    guided_state = torch.load(guided_path, map_location=device, weights_only=False)
    class_count = len(raw_state["class_names"])
    segmenter = UNet().to(device)
    segmenter.load_state_dict(segmentation_state["model_state"])
    raw_model = SmallCNN(num_classes=class_count).to(device)
    raw_model.load_state_dict(raw_state["model_state"])
    guided_model = SmallCNN(num_classes=class_count).to(device)
    guided_model.load_state_dict(guided_state["model_state"])
    return (
        segmenter,
        raw_model,
        guided_model,
        int(raw_state["image_size"]),
        float(guided_state.get("alpha", 0.25)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        default=Path("artifacts/segmentation/best.pt"),
    )
    parser.add_argument(
        "--raw-checkpoint",
        type=Path,
        default=Path("artifacts/classification/raw/best.pt"),
    )
    parser.add_argument(
        "--guided-checkpoint",
        type=Path,
        default=Path("artifacts/classification/guided/best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/robustness"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = _device(args.device)
    index = OxfordIIITPetIndex(args.root)
    segmenter, raw_model, guided_model, image_size, alpha = _load(
        args.segmentation_checkpoint,
        args.raw_checkpoint,
        args.guided_checkpoint,
        device,
    )
    records = index.records("test")
    if args.limit > 0:
        records = records[: args.limit]
    criterion = torch.nn.CrossEntropyLoss()
    result: dict[str, object] = {}
    for mode in ("original", "blur", "gray"):
        dataset = BackgroundDataset(index, records, image_size, mode)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        with torch.no_grad():
            raw = run_classification_epoch(
                raw_model,
                loader,
                device,
                criterion,
            )
            guided = run_classification_epoch(
                guided_model,
                loader,
                device,
                criterion,
                segmenter=segmenter,
                alpha=alpha,
            )
        result[mode] = {"raw": raw, "guided": guided}
        print(
            mode,
            "raw_acc=%.4f" % raw["accuracy"],
            "guided_acc=%.4f" % guided["accuracy"],
            "raw_f1=%.4f" % raw["macro_f1"],
            "guided_f1=%.4f" % guided["macro_f1"],
        )
    save_json(args.output / "metrics.json", result)


if __name__ == "__main__":
    main()
