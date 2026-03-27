"""
ais/models/gem_pool.py
----------------------
Generalized Mean (GeM) Pooling.

Standard in state-of-the-art image retrieval systems (used by Google, Meta,
national museum search engines). Learns the optimal pooling exponent `p`
during training rather than using a fixed mean or max.

Reference: Radenović et al., "Fine-tuning CNN Image Retrieval with No Human
Annotation", TPAMI 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean Pooling over spatial dimensions.

    Given feature map [B, C, H, W], returns [B, C] by computing:
        gem(x, p) = mean(x^p)^(1/p)

    p=1  → average pooling
    p=∞  → max pooling
    p≈3  → typical learned optimum for retrieval tasks

    Parameters
    ----------
    p : float
        Initial pooling exponent (learnable). Default 3.0.
    eps : float
        Clamp minimum to avoid numerical issues with negative values.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, C, H, W]  or  [B, N, C]  (ViT patch tokens)

        Returns
        -------
        [B, C]
        """
        if x.dim() == 3:
            # ViT patch tokens: [B, N, C] → treat N as spatial dim
            x = x.transpose(1, 2)              # [B, C, N]
            x = x.clamp(min=self.eps)
            x = x.pow(self.p)
            x = x.mean(dim=2)                  # [B, C]
            return x.pow(1.0 / self.p)
        else:
            # CNN feature map: [B, C, H, W]
            x = x.clamp(min=self.eps).pow(self.p)
            h, w = x.shape[2], x.shape[3]
            x = F.avg_pool2d(x, (h, w))        # [B, C, 1, 1]
            return x.pow(1.0 / self.p).squeeze(-1).squeeze(-1)  # [B, C]

    def extra_repr(self) -> str:
        return f"p={self.p.item():.4f}, eps={self.eps}"
