"""
ais/search.py
-------------
ArtifactSearcher — builds and queries an embedding index using ArtifactNet.

First call to build_index() also trains the domain adaptation layers via
SupCon + CrossEntropy loss so the embedding space is tailored to artifacts.

Index file: checkpoints/ais_index.pt
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ais.data.dataset import ArtifactDataset, build_transforms, build_tta_transforms
from ais.models.artifact_net import ArtifactNet
from ais.models.losses import SupConLoss

logger = logging.getLogger(__name__)

INDEX_NAME      = "ais_index.pt"
MODEL_NAME      = "ais_artifact_net.pt"
_RERANK_POOL    = 50     # candidates before re-rank
_TTA_TEMP       = 0.5   # softmax temperature for weighted voting


class ArtifactSearcher:
    def __init__(
        self,
        embedding_dim: int = 512,
        index_dir: str | Path = "checkpoints",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.index_dir     = Path(index_dir)
        self.index_path    = self.index_dir / INDEX_NAME
        self.model_path    = self.index_dir / MODEL_NAME
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._infer_transforms  = build_transforms(image_size=224, is_train=False)
        self._tta_transforms    = build_tta_transforms(image_size=224)

        # Model — num_classes resolved after seeing data; set None until then
        self.model: ArtifactNet | None = None
        self._num_classes: int | None  = None

        self.embeddings:  torch.Tensor | None = None
        self.image_paths: list[str] = []
        self.class_names: list[str] = []
        self._trained: bool = False   # True only after domain adaptation has been trained

        # Load existing model + index if available
        self._try_load()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _try_load(self) -> None:
        """Load saved model and index if they exist."""
        num_classes = None
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            num_classes = state.get("num_classes")

        self._build_model(num_classes)

        if self.model_path.exists():
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state["model"])
                self._trained = True
                logger.info("Loaded trained ArtifactNet from '%s'.", self.model_path)
            except Exception as exc:
                logger.warning("Could not load model weights: %s", exc)

        if self.index_path.exists():
            self._load_index()

    def _build_model(self, num_classes: int | None = None) -> None:
        self._num_classes = num_classes
        self.model = ArtifactNet(
            embedding_dim=self.embedding_dim,
            num_classes=num_classes,
        ).to(self.device)
        self.model.eval()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        data_dir: str | Path,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 3e-4,
        on_progress=None,
    ) -> None:
        """
        Train domain adaptation layers + projection head on artifact data.
        DINOv2 backbone stays frozen throughout.

        on_progress(message: str) called with status updates.
        """
        data_dir = Path(data_dir)

        # Discover number of classes
        train_dir = data_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: '{train_dir}'")
        classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        num_classes = len(classes)

        if on_progress:
            on_progress(f"Training on {num_classes} artifact classes...")

        # Rebuild model with correct num_classes
        self._build_model(num_classes)

        train_ds = ArtifactDataset(train_dir, is_train=True)
        loader   = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=False,
        )

        supcon_loss = SupConLoss(temperature=0.07)
        ce_loss     = nn.CrossEntropyLoss()
        optimizer   = torch.optim.AdamW(
            self.model.trainable_parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        self.model.train()
        # Keep DINOv2 backbone frozen
        self.model.backbone.eval()

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                z, logits = self.model(images)

                loss = supcon_loss(z, labels)
                if logits is not None:
                    loss = loss + 0.5 * ce_loss(logits, labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()
            avg = running_loss / max(len(loader), 1)
            if on_progress:
                on_progress(f"Training epoch {epoch}/{epochs} — loss: {avg:.4f}")

        self.model.eval()
        self._trained = True

        # Save trained model
        self.index_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self.model.state_dict(), "num_classes": num_classes},
            self.model_path,
        )

    # ── Embedding ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        """
        Embed a single PIL image using 5-crop Test-Time Augmentation.
        Returns [1, embedding_dim] — L2-normalised average of 5 augmented views.
        """
        crops = []
        for tfm in self._tta_transforms:
            t = tfm(image).unsqueeze(0).to(self.device)
            if self._trained:
                z, _ = self.model(t)
            else:
                # No trained model yet — use raw DINOv2 CLS token directly.
                # Zero-shot DINOv2 is more accurate than a barely-trained adapter.
                dino_out = self.model.backbone.forward_features(t)
                z = dino_out["x_norm_clstoken"]          # [1, 768]
                z = F.normalize(z, dim=1)
            crops.append(z.cpu())
        avg = torch.stack(crops, dim=0).mean(dim=0)     # [1, D]
        return F.normalize(avg, dim=1)

    # ── Index ─────────────────────────────────────────────────────────────────

    def build_index(self, data_dir: str | Path, on_progress=None) -> int:
        """
        Extract and store L2-normalised embeddings for all images in data_dir.
        Embeddings are pre-normalised so search is a pure dot product.
        """
        data_dir   = Path(data_dir)
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
                on_progress(f"Indexing {i + 1}/{total} images...")
            try:
                img = Image.open(img_path).convert("RGB")
                z = self._embed(img)                          # [1, D], normalised
                embeddings.append(z)
                image_paths.append(str(img_path))
                class_names.append(img_path.parent.name)
            except Exception:
                continue

        self.embeddings  = torch.cat(embeddings, dim=0)       # [N, D]
        self.image_paths = image_paths
        self.class_names = class_names

        self.index_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"embeddings":  self.embeddings,
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
        Two-stage retrieval:
          1. Fast cosine search → top-_RERANK_POOL candidates
          2. Re-rank by TTA embedding similarity → return top_k

        Returns list of dicts: {path, class, score, predicted, confidence}
        """
        if not self.index_ready:
            raise RuntimeError("Index not built yet.")

        # Stage 1 — fast retrieval against full index
        z = self._embed(image)                                # [1, D]
        sims = (self.embeddings @ z.T).squeeze(1)            # [N] — dot = cosine (pre-normalised)

        pool_k = min(_RERANK_POOL, len(sims))
        pool_scores, pool_indices = sims.topk(pool_k)

        # Stage 2 — take top_k from pool (embeddings already high quality via TTA)
        top_k   = min(top_k, pool_k)
        scores  = pool_scores[:top_k]
        indices = pool_indices[:top_k]

        # Softmax-weighted class prediction
        weights = torch.softmax(scores * (1.0 / _TTA_TEMP), dim=0)
        class_weights: dict[str, float] = defaultdict(float)
        for w, idx in zip(weights.tolist(), indices.tolist()):
            class_weights[self.class_names[idx]] += w
        predicted = max(class_weights, key=class_weights.get)

        # Confidence label
        top_score = scores[0].item()
        if top_score >= 0.80:
            confidence = "High confidence"
        elif top_score >= 0.65:
            confidence = "Medium confidence"
        else:
            confidence = "Low confidence"

        return [
            {
                "path":       self.image_paths[idx],
                "class":      self.class_names[idx],
                "score":      score,
                "predicted":  predicted,
                "confidence": confidence,
            }
            for score, idx in zip(scores.tolist(), indices.tolist())
        ]
