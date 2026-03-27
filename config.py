"""
config.py — central hyperparameter store for AIS.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
IMAGE_SIZE = 224

# ── Model ─────────────────────────────────────────────────────────────────────
EMBEDDING_DIM    = 512      # ArtifactNet output embedding dimension
NUM_CLASSES      = None     # resolved automatically from data folders
FREEZE_BACKBONE  = True     # DINOv2 backbone always frozen

# ── Training (domain adaptation layers only) ──────────────────────────────────
BATCH_SIZE    = 16
EPOCHS        = 10
LEARNING_RATE = 3e-4        # AdamW with cosine annealing

# ── Checkpoints ───────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_NAME = "ais_embedding_net.pt"   # kept for CLI train.py compatibility

# ── Firebase — shared cloud database ─────────────────────────────────────────
# Set FIREBASE_ENABLED = True and fill in the two values below after:
#   1. Creating a Firebase project at https://console.firebase.google.com
#   2. Enabling Firestore + Storage
#   3. Creating a Service Account → downloading the JSON key
FIREBASE_ENABLED     = False
FIREBASE_CREDENTIALS = "firebase-credentials.json"   # path to downloaded JSON key
FIREBASE_BUCKET      = ""                             # e.g. "your-project.appspot.com"

# ── Quick setup — first-time Analyze ─────────────────────────────────────────
# 100 images per class × 12 classes = 1,200 reference artifacts
# Concurrent downloads take ~2 minutes total.
QUICK_N_IMAGES = 100
QUICK_CLASSES = {
    "pottery":      "ancient pottery ceramic vessel",
    "coins":        "ancient coins numismatic currency",
    "weapons":      "ancient weapons sword spear bronze",
    "jewelry":      "ancient jewelry necklace bracelet gold",
    "sculpture":    "ancient sculpture figurine statuette",
    "tools":        "ancient tools implements flint stone",
    "vessels":      "ancient vessels amphora jug bowl",
    "inscriptions": "ancient inscriptions tablet cuneiform",
    "textiles":     "ancient textile fragment loom",
    "glass":        "ancient glass vessel Roman",
    "seals":        "ancient seals cylinder stamp",
    "mosaics":      "ancient mosaic tile floor",
}
