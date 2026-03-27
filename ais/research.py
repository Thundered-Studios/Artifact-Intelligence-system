"""
ais/research.py
---------------
Deep academic article search via the OpenAlex API (free, no key needed).
OpenAlex indexes 250M+ scholarly works including archaeology, art history,
and cultural heritage journals.

API docs: https://docs.openalex.org/
"""

from __future__ import annotations

import logging
from typing import Callable

import requests

logger = logging.getLogger(__name__)

_OPENALEX_URL = "https://api.openalex.org/works"
_HEADERS = {"User-Agent": "AIS/2.0 (Artifact Intelligence System; archaeology research)"}

# Keyword synonyms per class — used to boost text-based class scoring
CLASS_SYNONYMS: dict[str, list[str]] = {
    "pottery":      ["pottery", "ceramic", "earthenware", "terracotta", "faience",
                     "stoneware", "kiln", "fired clay", "amphora", "urn"],
    "coins":        ["coin", "numismatic", "currency", "medallion", "denarius",
                     "obol", "tetradrachm", "mint", "coinage", "monetary"],
    "weapons":      ["weapon", "sword", "spear", "dagger", "knife", "arrowhead",
                     "shield", "armor", "lance", "blade", "armament"],
    "jewelry":      ["jewelry", "jewel", "necklace", "bracelet", "ring", "earring",
                     "fibula", "brooch", "ornament", "pendant", "torque"],
    "sculpture":    ["sculpture", "figurine", "statuette", "relief", "bust",
                     "idol", "effigy", "terracotta figure", "votive"],
    "tools":        ["tool", "implement", "flint", "chisel", "hammer", "awl",
                     "scraper", "lithic", "knapping", "stone tool"],
    "vessels":      ["vessel", "amphora", "jug", "bowl", "cup", "flask",
                     "krater", "hydria", "oinochoe", "lekythos", "pyxis"],
    "inscriptions": ["inscription", "tablet", "cuneiform", "hieroglyph", "stele",
                     "papyrus", "writing", "script", "epigraphy", "palaeography"],
    "textiles":     ["textile", "fabric", "cloth", "loom", "weaving", "linen",
                     "wool", "embroidery", "tapestry", "thread"],
    "glass":        ["glass", "vitreous", "blown glass", "mosaic glass",
                     "unguentarium", "ungüentarium", "core-formed"],
    "seals":        ["seal", "cylinder seal", "stamp seal", "signet",
                     "impression", "glyptic", "intaglio", "engraved"],
    "mosaics":      ["mosaic", "tessera", "opus vermiculatum", "floor mosaic",
                     "wall mosaic", "tile", "tesserae"],
}


def text_class_boost(text: str, classes: list[str]) -> dict[str, float]:
    """
    Compute a boost multiplier for each class based on keyword overlap
    between the user's text description and class synonym lists.

    Returns dict {class_name: multiplier} where multiplier >= 1.0.
    Higher = more relevant to the text.
    """
    text_lower = text.lower()
    boosts: dict[str, float] = {}
    for cls in classes:
        synonyms = CLASS_SYNONYMS.get(cls, [cls.replace("_", " ")])
        hits = sum(1 for syn in synonyms if syn in text_lower)
        boosts[cls] = 1.0 + hits * 0.6    # each keyword hit adds 60% boost
    return boosts


def search_articles(
    query: str,
    max_results: int = 4,
    on_progress: Callable | None = None,
) -> list[dict]:
    """
    Search OpenAlex for academic articles relevant to `query`.

    Parameters
    ----------
    query       : search string (e.g. "ancient Roman coins numismatic")
    max_results : number of articles to return
    on_progress : optional callback(message: str)

    Returns list of dicts:
        {title, year, doi, url, abstract, cited_by, concepts}
    """
    if on_progress:
        on_progress("Searching academic literature...")

    params = {
        "search":   query,
        "per-page": max_results,
        "sort":     "cited_by_count:desc",   # most-cited first = most authoritative
        "select":   (
            "title,publication_year,doi,cited_by_count,"
            "abstract_inverted_index,concepts,primary_location"
        ),
    }
    try:
        resp = requests.get(
            _OPENALEX_URL, params=params, headers=_HEADERS, timeout=12
        )
        resp.raise_for_status()
        raw = resp.json().get("results", [])
    except requests.RequestException as exc:
        logger.warning("Article search failed: %s", exc)
        return []

    articles = []
    for r in raw:
        abstract = _reconstruct_abstract(r.get("abstract_inverted_index") or {})
        doi  = r.get("doi") or ""
        url  = doi if doi.startswith("http") else (f"https://doi.org/{doi}" if doi else "")
        concepts = [c["display_name"] for c in (r.get("concepts") or [])[:4]]

        articles.append({
            "title":    r.get("title") or "Untitled",
            "year":     r.get("publication_year"),
            "doi":      doi,
            "url":      url,
            "abstract": abstract[:400] if abstract else "(no abstract available)",
            "cited_by": r.get("cited_by_count", 0),
            "concepts": concepts,
        })

    return articles


def build_query(predicted_class: str, user_text: str = "") -> str:
    """Build an article search query from the predicted class and user description."""
    class_terms = CLASS_SYNONYMS.get(predicted_class, [predicted_class])[:3]
    base = f"ancient {predicted_class} {' '.join(class_terms[:2])} archaeological"
    if user_text.strip():
        base = f"{user_text.strip()} {base}"
    return base


def _reconstruct_abstract(inverted_index: dict) -> str:
    """OpenAlex stores abstracts as inverted index {word: [positions]}."""
    if not inverted_index:
        return ""
    words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))
