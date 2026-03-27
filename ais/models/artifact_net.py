"""
ais/models/artifact_net.py
--------------------------
ArtifactNet — government-grade artifact recognition model built from scratch.

Architecture
------------
Input [B, 3, 224, 224]
    │
DINOv2-ViT-B/14  (frozen — Meta 2023, trained on 142M images, best open features)
    │  CLS token [B, 768]  +  patch tokens [B, 256, 768]
    │
GeM Pooling over patch tokens → [B, 768]
    │
Fuse: concat(CLS, GeM) → [B, 1536]
    │
Domain Adaptation Block  (2× TransformerEncoderLayer, trained on artifact data)
    │  [B, 1536]
    │
Projection Head  (Linear → LayerNorm → GELU → Linear → L2-normalize)
    │
z  [B, embedding_dim=512]  ← final artifact embedding
    │
Auxiliary Classifier  (Linear → num_classes)
    │
logits  [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ais.models.gem_pool import GeM

# DINOv2 output dimension for ViT-B/14
_DINO_DIM = 768
_FUSED_DIM = _DINO_DIM * 2   # CLS + GeM patch concat


def _load_dinov2() -> nn.Module:
    """
    Load DINOv2-ViT-B/14 via torch.hub.
    Downloads ~330 MB on first call; cached locally thereafter.
    All parameters are frozen — DINOv2 needs no fine-tuning.
    """
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
        pretrained=True,
        verbose=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class DomainAdaptationBlock(nn.Module):
    """
    Two Transformer encoder layers that adapt DINOv2 features to the
    archaeological artifact domain. These are the only layers trained
    from scratch on artifact data.

    Input/output shape: [B, D]
    Internally treats the single vector as a sequence of length 1.
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] → unsqueeze to [B, 1, D] for Transformer
        x = x.unsqueeze(1)
        x = self.transformer(x)
        return x.squeeze(1)   # [B, D]


class ArtifactNet(nn.Module):
    """
    Parameters
    ----------
    embedding_dim : int
        Dimension of the output embedding z. Default 512.
    num_classes : int | None
        When provided, adds a linear classifier head. None = retrieval only.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()

        # ── DINOv2 backbone (frozen) ──────────────────────────────────────────
        self.backbone = _load_dinov2()

        # ── GeM over patch tokens ─────────────────────────────────────────────
        self.gem = GeM(p=3.0)

        # ── Domain adaptation (trained) ───────────────────────────────────────
        self.domain_adapt = DomainAdaptationBlock(d_model=_FUSED_DIM, nhead=8)

        # ── Projection head ───────────────────────────────────────────────────
        # [B, _FUSED_DIM] → [B, embedding_dim], L2-normalised
        self.projection = nn.Sequential(
            nn.Linear(_FUSED_DIM, _FUSED_DIM),
            nn.LayerNorm(_FUSED_DIM),
            nn.GELU(),
            nn.Linear(_FUSED_DIM, embedding_dim),
        )

        # ── Classifier head (optional) ────────────────────────────────────────
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
        x : [B, 3, H, W]  — ImageNet-normalised RGB

        Returns
        -------
        z      : [B, embedding_dim]  — L2-normalised embedding
        logits : [B, num_classes]    — or None
        """
        # DINOv2 returns dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
        with torch.no_grad():
            dino_out = self.backbone.forward_features(x)

        cls_token    = dino_out["x_norm_clstoken"]      # [B, 768]
        patch_tokens = dino_out["x_norm_patchtokens"]   # [B, N, 768]

        # GeM over patch tokens: [B, N, 768] → [B, 768]
        gem_feat = self.gem(patch_tokens)

        # Fuse CLS + GeM: [B, 1536]
        fused = torch.cat([cls_token, gem_feat], dim=1)

        # Domain adaptation: [B, 1536] → [B, 1536]
        adapted = self.domain_adapt(fused)

        # Project: [B, 1536] → [B, embedding_dim]
        z_raw = self.projection(adapted)

        # L2-normalise so cosine similarity = dot product
        z = F.normalize(z_raw, dim=1)

        # Classifier: [B, embedding_dim] → [B, num_classes]
        logits = self.classifier(z) if self.classifier is not None else None

        return z, logits

    # ── Helpers ───────────────────────────────────────────────────────────────

    def trainable_parameters(self):
        """Return only the parameters that should be trained (not DINOv2)."""
        return [p for p in self.parameters() if p.requires_grad]
