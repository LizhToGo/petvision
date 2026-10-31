"""Download Oxford-IIIT Pet images and prepare verified binary masks."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


OFFICIAL_URLS = {
    "images.tar.gz": (
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    ),
    "annotations.tar.gz": (
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    ),
}
ZENODO_LABELS_URL = (
    "https://zenodo.org/records/7116591/files/"
    "oxford_pets_labels.zip?download=1"
)
HF_REPO = "Shiym/ViT-FineTune"
HF_REVISION = "1cc46adc2014ae0f8d67cc55a84d2022886b0364"
BACKGROUND_COLOR = (68, 1, 84)
FOREGROUND_COLOR = (253, 231, 36)


def _download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        with destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        root = destination.resolve()
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        for member in members:
            tar.extract(member, destination, filter="data")


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as archive_file:
        root = destination.resolve()
        members = archive_file.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        archive_file.extractall(destination)


def _hf_image_archive() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename="oxford-iiit-pet/images.tar.gz",
            repo_type="dataset",
            revision=HF_REVISION,
        )
    )


def _hf_split_file(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"oxford-iiit-pet/annotations/{name}.txt",
            repo_type="dataset",
            revision=HF_REVISION,
        )
    )


def _convert_mask(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    foreground = np.all(rgb == FOREGROUND_COLOR, axis=2)
    background = np.all(rgb == BACKGROUND_COLOR, axis=2)
    if not np.all(foreground | background):
        raise RuntimeError(f"Unexpected segmentation colors in {source}")
    binary = np.where(foreground, 1, 2).astype(np.uint8)
    Image.fromarray(binary).save(destination)


def _filter_split_file(
    source: Path,
    destination: Path,
    valid_ids: set[str],
) -> None:
    kept: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image_id = line.split()[0]
        if image_id in valid_ids:
            kept.append(raw_line)
    if not kept:
        raise RuntimeError(f"No valid records remain in {source}")
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _zenodo_annotations(
    root: Path,
    downloads: Path,
    keep_archives: bool = False,
) -> None:
    archive = downloads / "oxford_pets_labels.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        if archive.exists():
            archive.unlink()
        _download(ZENODO_LABELS_URL, archive)
    else:
        print(f"Using existing archive {archive}")

    staging = root.parent / "zenodo-labels-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    _extract_zip(archive, staging)
    source = staging / "segmentations"
    if not source.is_dir():
        raise RuntimeError("Zenodo archive has no segmentations directory")

    trimap_dir = root / "annotations" / "trimaps"
    trimap_dir.mkdir(parents=True, exist_ok=True)
    valid_ids: set[str] = set()
    for source_mask in source.glob("*.png"):
        destination = trimap_dir / source_mask.name
        _convert_mask(source_mask, destination)
        valid_ids.add(source_mask.stem)
    if len(valid_ids) < 7000:
        raise RuntimeError(
            f"Zenodo annotations incomplete: found {len(valid_ids)} masks"
        )

    annotation_dir = root / "annotations"
    for split in ("trainval", "test"):
        source_split = _hf_split_file(split)
        _filter_split_file(
            source_split,
            annotation_dir / f"{split}.txt",
            valid_ids,
        )
    shutil.rmtree(staging)
    if not keep_archives:
        archive.unlink()


def _get_archive(filename: str, downloads: Path) -> Path:
    archive = downloads / filename
    if archive.exists():
        print(f"Using existing archive {archive}")
        return archive
    try:
        _download(OFFICIAL_URLS[filename], archive)
        return archive
    except Exception as exc:
        if archive.exists():
            archive.unlink()
        print(f"Official download failed: {exc}")
        if filename == "images.tar.gz":
            print("Using the pinned Hugging Face mirror for images.")
            return _hf_image_archive()
        raise


def _has_images(root: Path) -> bool:
    return (root / "images").is_dir() and sum(
        1 for _ in (root / "images").glob("*.jpg")
    ) >= 7000


def _has_annotations(root: Path) -> bool:
    trimap_dir = root / "annotations" / "trimaps"
    return (
        (root / "annotations" / "trainval.txt").is_file()
        and (root / "annotations" / "test.txt").is_file()
        and sum(1 for _ in trimap_dir.glob("*.png")) >= 7000
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/oxford-iiit-pet"),
        help="dataset directory containing images/ and annotations/",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="keep downloaded archives after extraction",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    downloads = root.parent / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    if _has_images(root):
        print(f"Using existing images under {root / 'images'}")
    else:
        image_archive = _get_archive("images.tar.gz", downloads)
        _extract_tar(image_archive, root)
        if not args.keep_archives and image_archive.parent == downloads:
            image_archive.unlink()

    if _has_annotations(root):
        print(f"Using existing annotations under {root / 'annotations'}")
    else:
        try:
            annotation_archive = _get_archive("annotations.tar.gz", downloads)
            _extract_tar(annotation_archive, root)
            if not args.keep_archives and annotation_archive.parent == downloads:
                annotation_archive.unlink()
        except Exception as exc:
            print(f"Official annotation download failed: {exc}")
            _zenodo_annotations(root, downloads, args.keep_archives)

    print(f"Dataset prepared at {root}")


if __name__ == "__main__":
    main()
