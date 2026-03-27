"""
ais/models/embedding_net.py
---------------------------
EmbeddingNet — ResNet18 backbone + projection head (+ optional classifier head).

Tensor shapes (B = batch size):
  Input  x  : [B, 3, 224, 224]   (ImageNet-normalised RGB)
  Output z   : [B, embedding_dim]  (L2-ready embedding)
  Output logits (optional): [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class EmbeddingNet(nn.Module):
    """
    Parameters
    ----------
    embedding_dim : int
        Size of the output embedding vector z (e.g. 256 or 512).
    num_classes : int | None
        When provided, adds a linear classifier head on top of z.
        Pass None to use the model as a pure embedding extractor.
    freeze_backbone : bool
        If True, the ResNet18 parameters are frozen so only the
        projection (and classifier) heads are trained.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        # Load ImageNet-pretrained ResNet18.
        # We drop its final fc layer (identity) so forward() returns a
        # 512-d pooled feature vector straight from the global average pool.
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone_out_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()                 # strip the original head
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Projection head ───────────────────────────────────────────────────
        # Maps 512-d backbone features → embedding_dim via a small MLP.
        # Shape: [B, 512] → [B, embedding_dim]
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_dim, embedding_dim),
        )

        # ── Classifier head (optional) ────────────────────────────────────────
        # Shape: [B, embedding_dim] → [B, num_classes]
        self.classifier: nn.Linear | None = None
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, 3, H, W]

        Returns
        -------
        z      : torch.Tensor  shape [B, embedding_dim]
        logits : torch.Tensor  shape [B, num_classes]  or  None
        """
        # [B, 3, H, W] → [B, 512]
        features = self.backbone(x)

        # [B, 512] → [B, embedding_dim]
        z = self.projection(features)

        # [B, embedding_dim] → [B, num_classes]  (or None)
        logits = self.classifier(z) if self.classifier is not None else None

        return z, logits
