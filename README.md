# AIS — Artifact Intelligence System

A cross-platform (Windows / macOS / Linux) PyTorch tool for archaeologists.
Given a photo of an artifact it produces:

- **z** — a compact embedding vector (e.g. 256-d) suitable for similarity search (FAISS).
- **logits** — optional classification scores for a configurable number of artifact types.

---

## Project layout

```
Artifact-Intelligence-system/
├── ais/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── embedding_net.py   # ResNet18 backbone + projection head
│   └── data/
│       ├── __init__.py
│       └── dataset.py         # ArtifactDataset (ImageFolder-style)
├── checkpoints/               # saved weights land here
├── data/                      # your images go here (see below)
│   ├── train/
│   │   └── <class_name>/
│   │       └── *.jpg
│   └── val/
│       └── <class_name>/
│           └── *.jpg
├── config.py                  # central hyperparameter store
├── train.py                   # training script
├── inference_demo.py          # single-image inference
├── requirements.txt
└── README.md
```

---

## 1 — Create a dummy dataset

The dataset loader expects one sub-folder per class under `data/train/` and `data/val/`.

**Linux / macOS**
```bash
mkdir -p data/train/pottery data/train/coins \
         data/val/pottery   data/val/coins
# Copy or download your images into those folders.
# For a smoke-test you can duplicate a single image:
cp path/to/any.jpg data/train/pottery/sample1.jpg
cp path/to/any.jpg data/val/pottery/sample1.jpg
cp path/to/any.jpg data/train/coins/sample1.jpg
cp path/to/any.jpg data/val/coins/sample1.jpg
```

**Windows (PowerShell)**
```powershell
New-Item -ItemType Directory -Force `
  data\train\pottery, data\train\coins, `
  data\val\pottery,   data\val\coins
# Then copy images into those folders.
```

---

## 2 — Create a virtual environment

```bash
# Works on all platforms (Python 3.10+)
python -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## 3 — Install requirements

> **PyTorch wheel:** visit https://pytorch.org/get-started/locally/ and select your
> OS / CUDA version to get the exact install command.
> CPU-only (any OS):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 4 — Train

```bash
python train.py \
    --data_dir    data \
    --epochs      5   \
    --batch_size  16  \
    --embedding_dim 256 \
    --num_classes 2
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Root directory containing `train/` and `val/` |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `16` | Images per batch |
| `--lr` | `1e-3` | Adam learning rate |
| `--embedding_dim` | `256` | Output embedding size |
| `--num_classes` | auto | Number of artifact classes (inferred from folders if omitted) |
| `--freeze_backbone` | `True` | Freeze ResNet18 weights; only train the heads |

Weights are saved to `checkpoints/ais_embedding_net.pt`.

---

## 5 — Run inference demo

```bash
python inference_demo.py \
    --image_path  path/to/artifact.jpg \
    --checkpoint  checkpoints/ais_embedding_net.pt \
    --embedding_dim 256 \
    --num_classes 2
```

The script prints the embedding shape, the first few values, predicted class probabilities
(if a classifier head was trained), and saves the embedding tensor to
`path/to/artifact.embedding.pt`.

---

## 6 — Extract embeddings for similarity search (FAISS-ready)

```python
import torch
from pathlib import Path
from PIL import Image

from ais.data.dataset import build_transforms
from ais.models.embedding_net import EmbeddingNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingNet(embedding_dim=256, num_classes=None)
model.load_state_dict(torch.load("checkpoints/ais_embedding_net.pt", map_location=device))
model.to(device).eval()

transform = build_transforms(image_size=224, is_train=False)

image_paths = list(Path("data/val").rglob("*.jpg"))
embeddings = []

with torch.no_grad():
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)   # [1, 3, 224, 224]
        z, _ = model(tensor)                              # [1, 256]
        embeddings.append(z.cpu())

# Stack into [N, 256] and save
all_embeddings = torch.cat(embeddings, dim=0)
torch.save(all_embeddings, "embeddings.pt")
print(f"Saved {all_embeddings.shape} embeddings to embeddings.pt")

# Load later for FAISS:
# import faiss, numpy as np
# index = faiss.IndexFlatL2(256)
# index.add(all_embeddings.numpy())
```

---

## Architecture overview

```
Input [B, 3, 224, 224]
        │
   ResNet18 (pretrained, fc removed)
        │
   [B, 512]  ← global average pool output
        │
   Linear(512→512) → ReLU → Linear(512→256)   ← projection head
        │
   z  [B, 256]                                  ← embedding output
        │
   Linear(256→num_classes)                      ← classifier head (optional)
        │
   logits [B, num_classes]
```
