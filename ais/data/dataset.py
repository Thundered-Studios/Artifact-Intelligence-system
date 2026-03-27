"""
ais/data/dataset.py
-------------------
ArtifactDataset — loads images from a directory tree and returns
(image_tensor, label_int) pairs.

Expected layout:
    <root>/
        pottery/   img001.jpg ...
        coins/     img002.jpg ...

Uses pathlib.Path throughout — no hard-coded separators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, is_train: bool = True) -> Callable:
    """Return a transform pipeline appropriate for training or inference."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.15),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def build_tta_transforms(image_size: int = 224) -> list[Callable]:
    """
    Five deterministic transforms for Test-Time Augmentation (TTA):
    center crop + four corner crops. Embeddings are averaged.
    """
    base = [
        transforms.Resize(image_size + 32),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
    crops = [
        transforms.CenterCrop(image_size),
        transforms.FiveCrop(image_size),   # returns tuple — handled in _embed_tta
    ]
    # Return 5 individual pipelines (center + 4 corners)
    corner_size = image_size
    pipelines = []
    # center
    pipelines.append(transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]))
    # four corners
    for corner_fn in [
        lambda img: img.crop((0, 0, corner_size, corner_size)),
        lambda img: img.crop((img.width - corner_size, 0, img.width, corner_size)),
        lambda img: img.crop((0, img.height - corner_size, corner_size, img.height)),
        lambda img: img.crop((img.width - corner_size, img.height - corner_size,
                              img.width, img.height)),
    ]:
        pipelines.append(transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.Lambda(corner_fn),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]))
    return pipelines


class ArtifactDataset(Dataset):
    """
    Parameters
    ----------
    root      : path to split directory, e.g. data/train or data/val
    image_size: spatial size after transforms
    is_train  : selects augmentation-enabled vs deterministic transforms
    """

    EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        is_train: bool = True,
    ) -> None:
        self.root = Path(root)
        self.transform = build_transforms(image_size, is_train)

        class_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not class_dirs:
            raise FileNotFoundError(
                f"No class sub-directories found under '{self.root}'."
            )

        self.classes: list[str] = [d.name for d in class_dirs]
        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        self.samples: list[tuple[Path, int]] = []
        seen: set[Path] = set()
        for cls_dir in class_dirs:
            label = self.class_to_idx[cls_dir.name]
            for ext in self.EXTENSIONS:
                for img_path in cls_dir.glob(f"*{ext}"):
                    if img_path not in seen:
                        self.samples.append((img_path, label))
                        seen.add(img_path)
                for img_path in cls_dir.glob(f"*{ext.upper()}"):
                    if img_path not in seen:
                        self.samples.append((img_path, label))
                        seen.add(img_path)

        if not self.samples:
            raise FileNotFoundError(f"No images found under '{self.root}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Returns (image_tensor [3, H, W], label int)."""
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label
