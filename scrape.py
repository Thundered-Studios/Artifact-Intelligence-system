"""
scrape.py
---------
CLI wrapper around ais.data.scraper.

Usage examples
--------------
# Default artifact classes (pottery, coins, weapons, jewelry, sculpture)
    python scrape.py

# Custom classes and query terms
    python scrape.py \
        --classes pottery="ancient pottery" coins="roman coins" \
        --n_images 150 \
        --out_dir data \
        --val_split 0.2

# Fewer images for a quick smoke-test
    python scrape.py --n_images 20
"""

from __future__ import annotations

import argparse
import logging

from ais.data.scraper import scrape_dataset

# ── Default classes shipped with AIS ─────────────────────────────────────────
DEFAULT_CLASSES: dict[str, str] = {
    "pottery":   "ancient pottery ceramic",
    "coins":     "ancient coins numismatic",
    "weapons":   "ancient weapons bronze iron",
    "jewelry":   "ancient jewelry ornament",
    "sculpture": "ancient sculpture figurine",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape artifact images from the Met Museum API."
    )
    p.add_argument(
        "--classes",
        nargs="*",
        metavar='NAME="QUERY"',
        help=(
            'Space-separated list of class=query pairs. '
            'Example: pottery="ancient pottery" coins="roman coins". '
            "Defaults to 5 built-in artifact categories."
        ),
    )
    p.add_argument(
        "--n_images",
        type=int,
        default=100,
        help="Number of images to download per class (default: 100).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="data",
        help="Root output directory (default: data).",
    )
    p.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of images held out for validation (default: 0.2).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split (default: 42).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


def _parse_class_args(raw: list[str]) -> dict[str, str]:
    """
    Parse ['pottery=ancient pottery', 'coins=roman coins'] into
    {'pottery': 'ancient pottery', 'coins': 'roman coins'}.
    """
    classes: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(
                f"Invalid class spec '{item}'. "
                "Expected format: class_name=search query"
            )
        name, _, query = item.partition("=")
        classes[name.strip()] = query.strip().strip('"').strip("'")
    return classes


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    classes = (
        _parse_class_args(args.classes)
        if args.classes
        else DEFAULT_CLASSES
    )

    print("AIS Scraper — Metropolitan Museum of Art")
    print(f"  Classes    : {list(classes.keys())}")
    print(f"  Images/cls : {args.n_images}")
    print(f"  Val split  : {args.val_split}")
    print(f"  Output dir : {args.out_dir}")
    print()

    scrape_dataset(
        classes=classes,
        out_dir=args.out_dir,
        n_images=args.n_images,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
