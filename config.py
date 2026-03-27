"""
config.py — central hyperparameter store for AIS.
All values can be overridden via argparse in train.py / inference_demo.py.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = "data"          # root that contains train/ and val/ sub-folders
IMAGE_SIZE = 224           # resize both sides to this before centre-crop

# ── Model ─────────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 256        # dimension of the output embedding vector z
NUM_CLASSES = None         # set to an int (e.g. 5) to add a classifier head
FREEZE_BACKBONE = True     # freeze ResNet weights during the first training run

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3

# ── Checkpoints ───────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "ais_embedding_net.pt"

# ── Quick setup (first-time Analyze) ──────────────────────────────────────────
# Images scraped per class on first run. Small = fast setup (~20-30 seconds).
QUICK_N_IMAGES = 15
QUICK_CLASSES = {
    "pottery":   "ancient pottery ceramic",
    "coins":     "ancient coins numismatic",
    "weapons":   "ancient weapons bronze",
    "jewelry":   "ancient jewelry ornament",
    "sculpture": "ancient sculpture figurine",
}
