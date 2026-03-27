"""
ais/data/scraper.py
-------------------
Downloads artifact images from the Metropolitan Museum of Art's free
public API (no API key required) and organises them into the
train/val folder structure expected by ArtifactDataset.

API docs: https://metmuseum.github.io/

Output layout
-------------
<out_dir>/
    train/
        <class_name>/
            <objectID>.jpg
            ...
    val/
        <class_name>/
            <objectID>.jpg
            ...
"""

from __future__ import annotations

import time
import random
import logging
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Met Museum API endpoints ──────────────────────────────────────────────────
_SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
_OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

# Be polite to the API — wait between requests
_REQUEST_DELAY = 0.3   # seconds between object-detail calls
_DOWNLOAD_DELAY = 0.1  # seconds between image downloads


def _search_object_ids(query: str, max_results: int = 500) -> list[int]:
    """Return up to `max_results` object IDs matching `query`."""
    params = {"q": query, "hasImages": "true"}
    try:
        resp = requests.get(_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Search failed for '%s': %s", query, exc)
        return []

    data = resp.json()
    ids: list[int] = data.get("objectIDs") or []
    return ids[:max_results]


def _fetch_primary_image_url(object_id: int) -> str | None:
    """Return the primaryImage URL for an object, or None if unavailable."""
    try:
        resp = requests.get(_OBJECT_URL.format(object_id), timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Object %s fetch failed: %s", object_id, exc)
        return None

    data = resp.json()
    url = data.get("primaryImage", "")
    return url if url else None


def _download_image(url: str, dest: Path) -> bool:
    """Download `url` to `dest`. Returns True on success."""
    try:
        resp = requests.get(url, timeout=20, stream=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as exc:
        logger.debug("Image download failed (%s): %s", url, exc)
        return False


def scrape_class(
    query: str,
    class_name: str,
    out_dir: Path,
    n_images: int = 100,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    """
    Search the Met API for `query`, download up to `n_images` images,
    and split them into train/val folders under `out_dir/<split>/<class_name>/`.

    Parameters
    ----------
    query      : search term sent to the Met API (e.g. "ancient pottery")
    class_name : folder name used as the class label (e.g. "pottery")
    out_dir    : root output directory (e.g. Path("data"))
    n_images   : how many images to download for this class
    val_split  : fraction held out for validation (0–1)
    seed       : random seed for reproducible train/val split

    Returns
    -------
    {"train": n_train, "val": n_val}  — counts of successfully saved images
    """
    train_dir = out_dir / "train" / class_name
    val_dir   = out_dir / "val"   / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Searching Met API for '%s'…", query)
    object_ids = _search_object_ids(query, max_results=n_images * 5)

    if not object_ids:
        logger.warning("No results for query '%s'. Skipping.", query)
        return {"train": 0, "val": 0}

    # Shuffle for variety; reproducible with seed
    rng = random.Random(seed)
    rng.shuffle(object_ids)

    # Collect image URLs (up to n_images usable ones)
    usable: list[tuple[int, str]] = []
    for oid in tqdm(object_ids, desc=f"  Resolving '{class_name}' URLs", leave=False):
        if len(usable) >= n_images:
            break
        time.sleep(_REQUEST_DELAY)
        url = _fetch_primary_image_url(oid)
        if url:
            usable.append((oid, url))

    if not usable:
        logger.warning("No downloadable images found for '%s'.", query)
        return {"train": 0, "val": 0}

    # Train / val split
    n_val   = max(1, int(len(usable) * val_split))
    val_set = set(oid for oid, _ in usable[:n_val])

    counts = {"train": 0, "val": 0}
    for oid, url in tqdm(usable, desc=f"  Downloading '{class_name}'"):
        split    = "val" if oid in val_set else "train"
        dest_dir = val_dir if split == "val" else train_dir
        dest     = dest_dir / f"{oid}.jpg"

        if dest.exists():           # skip already-downloaded files
            counts[split] += 1
            continue

        time.sleep(_DOWNLOAD_DELAY)
        if _download_image(url, dest):
            counts[split] += 1
        else:
            logger.debug("Skipped object %s (download error).", oid)

    logger.info(
        "  '%s' done — train: %d  val: %d",
        class_name, counts["train"], counts["val"],
    )
    return counts


def scrape_dataset(
    classes: dict[str, str],
    out_dir: str | Path = "data",
    n_images: int = 100,
    val_split: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Scrape multiple artifact classes in sequence.

    Parameters
    ----------
    classes   : mapping of {class_name: search_query}
                e.g. {"pottery": "ancient pottery", "coins": "ancient coins"}
    out_dir   : root data directory (default "data")
    n_images  : images to download per class
    val_split : fraction held out for validation
    seed      : RNG seed
    """
    out_dir = Path(out_dir)
    total = {"train": 0, "val": 0}

    for class_name, query in classes.items():
        print(f"\n[{class_name}] query='{query}'")
        counts = scrape_class(
            query=query,
            class_name=class_name,
            out_dir=out_dir,
            n_images=n_images,
            val_split=val_split,
            seed=seed,
        )
        total["train"] += counts["train"]
        total["val"]   += counts["val"]

    print(
        f"\nDataset ready in '{out_dir}' — "
        f"train: {total['train']} images  val: {total['val']} images"
    )
