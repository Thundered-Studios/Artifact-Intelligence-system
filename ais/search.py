"""
ais/search.py
-------------
Builds and queries an embedding index for similarity search.
Uses pretrained ResNet18 — no prior training required.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from ais.data.dataset import build_transforms

INDEX_NAME = "ais_index.pt"


def _build_backbone() -> tuple[nn.Module, int]:
    """Load pretrained ResNet18, strip the classifier, return (model, feat_dim)."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feat_dim = backbone.fc.in_features   # 512
    backbone.fc = nn.Identity()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone, feat_dim


class ArtifactSearcher:
    def __init__(self, index_dir: str | Path = "checkpoints") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_transforms(image_size=224, is_train=False)
        self.index_path = Path(index_dir) / INDEX_NAME

        self._backbone, self._feat_dim = _build_backbone()
        self._backbone.to(self.device)

        self.embeddings:  torch.Tensor | None = None
        self.image_paths: list[str] = []
        self.class_names: list[str] = []

        if self.index_path.exists():
            self._load_index()

    # ── Embedding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        """Return [1, feat_dim] embedding for a PIL image."""
        t = self.transform(image).unsqueeze(0).to(self.device)
        return self._backbone(t).cpu()

    # ── Index ─────────────────────────────────────────────────────────────────

    def build_index(self, data_dir: str | Path, on_progress=None) -> int:
        """
        Walk data_dir (train + val), embed every image, save index.
        on_progress(message) called with status strings.
        Returns number of images indexed.
        """
        data_dir = Path(data_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        all_paths: list[Path] = []
        for split in ("train", "val"):
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for f in cls_dir.iterdir():
                    if f.suffix.lower() in extensions:
                        all_paths.append(f)

        if not all_paths:
            raise FileNotFoundError(f"No images found under '{data_dir}'.")

        embeddings, image_paths, class_names = [], [], []
        total = len(all_paths)

        for i, img_path in enumerate(all_paths):
            if on_progress:
                on_progress(f"Indexing image {i + 1} of {total}...")
            try:
                img = Image.open(img_path).convert("RGB")
                z = self._embed(img)
                embeddings.append(z)
                image_paths.append(str(img_path))
                class_names.append(img_path.parent.name)
            except Exception:
                continue

        self.embeddings  = torch.cat(embeddings, dim=0)
        self.image_paths = image_paths
        self.class_names = class_names

        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"embeddings": self.embeddings,
             "image_paths": self.image_paths,
             "class_names": self.class_names},
            self.index_path,
        )
        return len(image_paths)

    def _load_index(self) -> None:
        data = torch.load(self.index_path, map_location="cpu")
        self.embeddings  = data["embeddings"]
        self.image_paths = data["image_paths"]
        self.class_names = data["class_names"]

    @property
    def index_ready(self) -> bool:
        return self.embeddings is not None and len(self.embeddings) > 0

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, image: Image.Image, top_k: int = 6) -> list[dict]:
        """
        Return top_k most similar artifacts.
        Each result: {"path": str, "class": str, "score": float}
        """
        if not self.index_ready:
            raise RuntimeError("Index not built yet.")

        z = self._embed(image)
        q  = nn.functional.normalize(z, dim=1)
        db = nn.functional.normalize(self.embeddings, dim=1)
        sims = (db @ q.T).squeeze(1)

        top_k = min(top_k, len(sims))
        scores, indices = sims.topk(top_k)

        # Majority-vote class from top results
        top_classes = [self.class_names[i] for i in indices.tolist()]
        predicted = max(set(top_classes), key=top_classes.count)

        return [
            {
                "path":      self.image_paths[idx],
                "class":     self.class_names[idx],
                "score":     score,
                "predicted": predicted,
            }
            for score, idx in zip(scores.tolist(), indices.tolist())
        ]
