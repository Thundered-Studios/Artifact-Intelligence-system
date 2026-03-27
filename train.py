"""
train.py
--------
Minimal training script for AIS EmbeddingNet.

Usage
-----
    python train.py --data_dir data --epochs 5 --batch_size 16 --num_classes 5
"""

from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from ais.data.dataset import ArtifactDataset
from ais.models.embedding_net import EmbeddingNet


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AIS EmbeddingNet")
    p.add_argument("--data_dir",      type=str, default=config.DATA_DIR)
    p.add_argument("--epochs",        type=int, default=config.EPOCHS)
    p.add_argument("--batch_size",    type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr",            type=float, default=config.LEARNING_RATE)
    p.add_argument("--embedding_dim", type=int, default=config.EMBEDDING_DIM)
    p.add_argument("--num_classes",   type=int, default=config.NUM_CLASSES)
    p.add_argument("--freeze_backbone", action="store_true",
                   default=config.FREEZE_BACKBONE)
    p.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA if available, else CPU — works on all platforms."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> tuple[float, float]:
    """Return (avg_loss, accuracy). accuracy=0 when no classifier exists."""
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            z, logits = model(images)
            if logits is not None and criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
            total += images.size(0)
    avg_loss = total_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    # pathlib.Path handles Windows back-slashes and Unix forward-slashes alike.
    data_root = pathlib.Path(args.data_dir)
    train_ds = ArtifactDataset(data_root / "train", is_train=True)
    val_ds   = ArtifactDataset(data_root / "val",   is_train=False)

    print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # num_workers=0 avoids multiprocessing pickling issues on Windows.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = args.num_classes or len(train_ds.classes)
    model = EmbeddingNet(
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # ── Optimiser & loss ──────────────────────────────────────────────────────
    # Only optimise parameters that require grad (projection + classifier heads
    # when backbone is frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss() if num_classes else None

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # cross-platform mkdir

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # images : [B, 3, 224, 224]

            optimizer.zero_grad()
            z, logits = model(images)
            # z      : [B, embedding_dim]
            # logits : [B, num_classes] or None

            if logits is not None and criterion is not None:
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                # No classifier — nothing to back-prop yet; skip update.
                pass

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch}/{args.epochs}  "
                    f"Batch {batch_idx+1}/{len(train_loader)}  "
                    f"Loss: {running_loss / (batch_idx + 1):.4f}"
                )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )

    # ── Save weights ──────────────────────────────────────────────────────────
    ckpt_path = ckpt_dir / config.CHECKPOINT_NAME
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nModel saved to '{ckpt_path}'")


if __name__ == "__main__":
    main()
