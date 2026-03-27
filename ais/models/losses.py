"""
ais/models/losses.py
--------------------
Supervised Contrastive Loss (SupCon).

Shapes the embedding space so that images of the same artifact class cluster
tightly together while different classes are pushed apart. Far more powerful
than CrossEntropy for retrieval tasks.

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
https://arxiv.org/abs/2004.11362
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Parameters
    ----------
    temperature : float
        Logit scaling factor. Lower = sharper distribution. Default 0.07.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : [B, D]  — L2-normalised embeddings
        labels   : [B]     — integer class ids

        Returns
        -------
        scalar loss
        """
        device = features.device
        B = features.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Ensure L2-normalised
        features = F.normalize(features, dim=1)

        # Similarity matrix [B, B], scaled by temperature
        sim = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # Mask: 1 where same class (excluding diagonal)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float().to(device)             # [B, B]
        mask_self = torch.eye(B, device=device)
        mask_pos = mask_pos - mask_self                                 # exclude self

        # For numerical stability subtract max per row
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Log-sum-exp over all negatives (entire row minus self)
        exp_sim = torch.exp(sim) * (1 - mask_self)                     # [B, B]
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # Average log-prob over positive pairs
        n_pos = mask_pos.sum(dim=1)                                     # [B]
        # Skip rows with no positives (only-child classes in a batch)
        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(mask_pos * log_prob).sum(dim=1)
        loss = (loss[valid] / n_pos[valid]).mean()
        return loss
