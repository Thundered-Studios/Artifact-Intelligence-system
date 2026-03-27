"""
ais/search.py
-------------
Builds and queries an in-memory embedding index for similarity search.
No FAISS required — uses cosine similarity via PyTorch.

Index file: checkpoints/ais_index.pt
  {
    "embeddings":   Tensor [N, embedding_dim],
    "image_paths":  list[str],
    "class_names":  list[str],
  }
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ais.data.dataset import build_transforms
from ais.models.embedding_net import EmbeddingNet


class ArtifactSearcher:
    INDEX_NAME = "ais_index.pt"

    def __init__(
        self,
        checkpoint: str | Path,
        embedding_dim: int = 256,
        num_classes: int | None = None,
        index_dir: str | Path = "checkpoints",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_transforms(image_size=224, is_train=False)
        self.index_path = Path(index_dir) / self.INDEX_NAME

        # ── Load model ────────────────────────────────────────────────────────
        self.model = EmbeddingNet(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            freeze_backbone=False,
        )
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        # ── Load index if it exists ───────────────────────────────────────────
        self.embeddings: torch.Tensor | None = None
        self.image_paths: list[str] = []
        self.class_names: list[str] = []

        if self.index_path.exists():
            self._load_index()

    # ── Index management ──────────────────────────────────────────────────────

    def build_index(self, data_dir: str | Path, on_progress=None) -> int:
        """
        Walk `data_dir` (train + val), embed every image, save index.
        Calls on_progress(current, total) if provided.
        Returns the number of images indexed.
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

        with torch.no_grad():
            for i, img_path in enumerate(all_paths):
                if on_progress:
                    on_progress(i + 1, total)
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = self.transform(img).unsqueeze(0).to(self.device)
                    z, _ = self.model(tensor)          # [1, embedding_dim]
                    embeddings.append(z.cpu())
                    image_paths.append(str(img_path))
                    class_names.append(img_path.parent.name)
                except Exception:
                    continue

        self.embeddings = torch.cat(embeddings, dim=0)   # [N, embedding_dim]
        self.image_paths = image_paths
        self.class_names = class_names

        torch.save(
            {
                "embeddings":  self.embeddings,
                "image_paths": self.image_paths,
                "class_names": self.class_names,
            },
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

    def search(
        self, image: Image.Image, top_k: int = 6
    ) -> list[dict]:
        """
        Find the top_k most similar artifacts to `image`.

        Returns list of dicts:
            [{"path": str, "class": str, "score": float}, ...]
        sorted by descending similarity.
        """
        if not self.index_ready:
            raise RuntimeError("Index not built. Call build_index() first.")

        with torch.no_grad():
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            z, logits = self.model(tensor)             # [1, embedding_dim]

        # Cosine similarity between query and every indexed embedding
        q = torch.nn.functional.normalize(z.cpu(), dim=1)           # [1, D]
        db = torch.nn.functional.normalize(self.embeddings, dim=1)  # [N, D]
        sims = (db @ q.T).squeeze(1)                                 # [N]

        top_k = min(top_k, len(sims))
        scores, indices = sims.topk(top_k)

        # Predicted class from classifier head (if present)
        predicted_class: str | None = None
        if logits is not None:
            predicted_idx = logits.argmax(dim=1).item()
            # Map index back to class name via the index's unique classes
            unique_classes = sorted(set(self.class_names))
            if predicted_idx < len(unique_classes):
                predicted_class = unique_classes[predicted_idx]

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append({
                "path":      self.image_paths[idx],
                "class":     self.class_names[idx],
                "score":     score,
                "predicted": predicted_class,
            })

        return results
