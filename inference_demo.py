"""
inference_demo.py
-----------------
Load a saved checkpoint and run a single image through EmbeddingNet.

Usage
-----
    python inference_demo.py \
        --image_path  path/to/artifact.jpg \
        --checkpoint  checkpoints/ais_embedding_net.pt \
        --embedding_dim 256 \
        --num_classes 5
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from PIL import Image

import config
from ais.data.dataset import build_transforms
from ais.models.embedding_net import EmbeddingNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AIS inference demo")
    p.add_argument("--image_path",    type=str, required=True)
    p.add_argument("--checkpoint",    type=str,
                   default=str(pathlib.Path(config.CHECKPOINT_DIR) / config.CHECKPOINT_NAME))
    p.add_argument("--embedding_dim", type=int, default=config.EMBEDDING_DIM)
    p.add_argument("--num_classes",   type=int, default=config.NUM_CLASSES)
    p.add_argument("--image_size",    type=int, default=config.IMAGE_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = EmbeddingNet(
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        freeze_backbone=False,   # doesn't matter at inference time
    )
    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}'")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded checkpoint: '{ckpt_path}'")

    # ── Load & transform image ────────────────────────────────────────────────
    img_path = pathlib.Path(args.image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: '{img_path}'")

    transform = build_transforms(args.image_size, is_train=False)
    image = Image.open(img_path).convert("RGB")
    # transform returns [3, H, W]; unsqueeze(0) adds batch dim → [1, 3, H, W]
    tensor = transform(image).unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        z, logits = model(tensor)
    # z      : [1, embedding_dim]
    # logits : [1, num_classes] or None

    print(f"\nEmbedding shape : {tuple(z.shape)}")
    print(f"Embedding (first 8 values): {z[0, :8].cpu().tolist()}")

    if logits is not None:
        probs = torch.softmax(logits, dim=1)[0]
        predicted_class = probs.argmax().item()
        print(f"\nClassifier output:")
        print(f"  Predicted class index : {predicted_class}")
        print(f"  Class probabilities   : {[round(p, 4) for p in probs.tolist()]}")
    else:
        print("\nNo classifier head — embedding-only mode.")

    # ── Optional: save embedding to disk ─────────────────────────────────────
    # Embeddings are saved as a .pt tensor for later FAISS similarity search.
    out_path = img_path.with_suffix(".embedding.pt")
    torch.save(z.cpu(), out_path)
    print(f"\nEmbedding saved to '{out_path}' (shape {tuple(z.shape)})")


if __name__ == "__main__":
    main()
