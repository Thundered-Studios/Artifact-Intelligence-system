"""
ais/feedback.py
---------------
Correction feedback store — lets archaeologists flag wrong predictions
so the model can learn from them.

Each correction stores:
  - The image embedding (so we can use it for retraining)
  - The wrong predicted class
  - The correct class

When enough corrections accumulate the app offers to retrain.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

CORRECTIONS_FILE = "corrections.json"
RETRAIN_THRESHOLD = 10   # suggest retraining after this many corrections


class FeedbackStore:
    """
    Persistent store for user corrections.

    Saved to: <store_dir>/corrections.json
    """

    def __init__(self, store_dir: str | Path = "checkpoints") -> None:
        self.path = Path(store_dir) / CORRECTIONS_FILE
        self._corrections: list[dict] = self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        embedding: torch.Tensor,
        predicted: str,
        correct: str,
    ) -> None:
        """
        Record that the model predicted `predicted` but the correct class
        is `correct`. `embedding` is the [1, D] query embedding.
        """
        self._corrections.append({
            "embedding": embedding.squeeze(0).tolist(),
            "predicted": predicted,
            "correct":   correct,
        })
        self._save()
        logger.info(
            "Correction saved: predicted=%s correct=%s (total=%d)",
            predicted, correct, len(self._corrections),
        )

    def __len__(self) -> int:
        return len(self._corrections)

    @property
    def should_suggest_retrain(self) -> bool:
        return len(self) >= RETRAIN_THRESHOLD

    def as_tensors(self) -> tuple[torch.Tensor, list[str]]:
        """
        Return (embeddings [N, D], correct_classes [N]) for retraining.
        """
        if not self._corrections:
            return torch.zeros(0), []
        embeddings = torch.tensor([c["embedding"] for c in self._corrections])
        labels = [c["correct"] for c in self._corrections]
        return embeddings, labels

    def clear(self) -> None:
        """Clear all stored corrections (call after successful retraining)."""
        self._corrections = []
        self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not load corrections: %s", exc)
        return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._corrections, f)
