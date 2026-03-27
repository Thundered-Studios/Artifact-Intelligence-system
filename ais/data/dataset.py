"""
ais/data/dataset.py
-------------------
ArtifactDataset — loads images from a directory tree and returns
(image_tensor, label_int) pairs.

Expected layout (same convention as torchvision.datasets.ImageFolder):

    <root>/
        train/
            pottery/
                img001.jpg
                img002.jpg
            coins/
                img003.jpg
        val/
            pottery/
                img004.jpg
            coins/
                img005.jpg

Uses pathlib.Path throughout — no hard-coded path separators,
so the code is identical on Windows, macOS, and Linux.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── Standard ImageNet normalisation constants ─────────────────────────────────
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.224, 0.225, 0.225)


def build_transforms(image_size: int = 224, is_train: bool = True) -> Callable:
    """Return a torchvision transform pipeline."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize(image_size + 32),   # a bit larger, then crop
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class ArtifactDataset(Dataset):
    """
    Parameters
    ----------
    root : str | Path
        Path to the split directory, e.g. ``data/train`` or ``data/val``.
    image_size : int
        Both spatial dimensions are resized to this value.
    is_train : bool
        Selects augmentation-enabled (True) vs. deterministic (False) transforms.
    extensions : tuple[str, ...]
        File suffixes considered as images (case-insensitive).
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

        # ── Discover classes (one sub-folder = one class) ─────────────────────
        # sorted() gives a deterministic, OS-independent ordering.
        class_dirs = sorted(
            p for p in self.root.iterdir() if p.is_dir()
        )
        if not class_dirs:
            raise FileNotFoundError(
                f"No class sub-directories found under '{self.root}'. "
                "Expected layout: <root>/<class_name>/*.jpg"
            )

        self.classes: list[str] = [d.name for d in class_dirs]
        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # ── Collect (image_path, label) pairs ────────────────────────────────
        self.samples: list[tuple[Path, int]] = []
        for cls_dir in class_dirs:
            label = self.class_to_idx[cls_dir.name]
            for ext in self.EXTENSIONS:
                # glob is case-sensitive on Linux; handle upper/lower manually
                for img_path in cls_dir.glob(f"*{ext}"):
                    self.samples.append((img_path, label))
                for img_path in cls_dir.glob(f"*{ext.upper()}"):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No image files found under '{self.root}'. "
                f"Supported extensions: {self.EXTENSIONS}"
            )

    # ── Dataset protocol ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        image : torch.Tensor  shape [3, image_size, image_size]
        label : int
        """
        img_path, label = self.samples[idx]
        # Always convert to RGB so grayscale / RGBA images work uniformly.
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
