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

import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Met Museum API endpoints ──────────────────────────────────────────────────
_SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
_OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

_CONCURRENT_WORKERS = 8   # parallel threads for URL resolution + image download


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
    on_progress=None,
) -> dict[str, int]:
    """
    Search the Met API for `query`, download up to `n_images` images
    concurrently, and split into train/val folders.

    on_progress(message: str) is called with status updates when provided.
    """
    train_dir = out_dir / "train" / class_name
    val_dir   = out_dir / "val"   / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if on_progress:
        on_progress(f"Searching for {class_name}...")

    object_ids = _search_object_ids(query, max_results=n_images * 4)
    if not object_ids:
        logger.warning("No results for query '%s'. Skipping.", query)
        return {"train": 0, "val": 0}

    rng = random.Random(seed)
    rng.shuffle(object_ids)
    candidates = object_ids[: n_images * 3]   # fetch 3× to account for misses

    # ── Resolve image URLs concurrently ───────────────────────────────────────
    if on_progress:
        on_progress(f"Resolving {class_name} image URLs...")

    usable: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as pool:
        futures = {pool.submit(_fetch_primary_image_url, oid): oid for oid in candidates}
        for future in as_completed(futures):
            if len(usable) >= n_images:
                break
            oid = futures[future]
            url = future.result()
            if url:
                usable.append((oid, url))

    usable = usable[:n_images]
    if not usable:
        return {"train": 0, "val": 0}

    # ── Train / val split ─────────────────────────────────────────────────────
    n_val   = max(1, int(len(usable) * val_split))
    val_set = set(oid for oid, _ in usable[:n_val])

    # ── Download images concurrently ──────────────────────────────────────────
    if on_progress:
        on_progress(f"Downloading {class_name} images...")

    counts = {"train": 0, "val": 0}

    def _fetch_one(item):
        oid, url = item
        split    = "val" if oid in val_set else "train"
        dest_dir = val_dir if split == "val" else train_dir
        dest     = dest_dir / f"{oid}.jpg"
        if dest.exists():
            return split
        return split if _download_image(url, dest) else None

    with ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as pool:
        for result in pool.map(_fetch_one, usable):
            if result:
                counts[result] += 1

    return counts


def scrape_dataset(
    classes: dict[str, str],
    out_dir: str | Path = "data",
    n_images: int = 100,
    val_split: float = 0.2,
    seed: int = 42,
    on_progress=None,
) -> int:
    """
    Scrape multiple artifact classes in sequence.
    Returns total number of images downloaded.
    on_progress(message: str) called with status updates.
    """
    out_dir = Path(out_dir)
    total = 0

    for class_name, query in classes.items():
        counts = scrape_class(
            query=query,
            class_name=class_name,
            out_dir=out_dir,
            n_images=n_images,
            val_split=val_split,
            seed=seed,
            on_progress=on_progress,
        )
        total += counts["train"] + counts["val"]

    return total
