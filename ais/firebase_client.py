"""
ais/firebase_client.py
----------------------
Firebase integration for AIS — shared cloud artifact database.

Provides:
  - Cloud embedding index (Firebase Storage) shared across all users
  - Artifact metadata (Firestore) growing with every contribution
  - Correction sync (Firestore) for community-driven model improvement
  - Lazy image download for result thumbnails not cached locally

Firestore schema
----------------
index_versions/current
    version:         str        e.g. "v2"
    updated_at:      timestamp
    artifact_counts: dict       {class: count}
    total:           int

artifacts/{auto_id}
    class:           str
    source:          str        "met_museum" | "user"
    image_url:       str        original URL
    storage_path:    str        Firebase Storage path
    created_at:      timestamp

corrections/{auto_id}
    predicted:       str
    correct:         str
    embedding:       list[float]
    created_at:      timestamp

Firebase Storage layout
-----------------------
embeddings/{version}/{class_name}.pt   — [N, 768] tensor per class
metadata/{version}/{class_name}.json   — [{image_url, storage_path, source}, ...]
images/{class_name}/{filename}.jpg     — artifact images
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Callable

import torch

logger = logging.getLogger(__name__)

_app_initialized = False


def _init_firebase(credentials_path: str, bucket_name: str) -> bool:
    """Initialize Firebase app. Returns True on success."""
    global _app_initialized
    if _app_initialized:
        return True
    try:
        import firebase_admin
        from firebase_admin import credentials as fb_creds

        if not firebase_admin._apps:
            cred = fb_creds.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
        _app_initialized = True
        return True
    except Exception as exc:
        logger.warning("Firebase init failed: %s", exc)
        return False


class FirebaseClient:
    """
    Wraps Firestore + Firebase Storage for AIS cloud sync.

    All methods fail gracefully — if Firebase is unavailable the app
    continues in offline mode.
    """

    def __init__(self, credentials_path: str, bucket_name: str) -> None:
        self._ok = _init_firebase(credentials_path, bucket_name)
        self._db     = None
        self._bucket = None
        if self._ok:
            try:
                from firebase_admin import firestore, storage
                self._db     = firestore.client()
                self._bucket = storage.bucket()
            except Exception as exc:
                logger.warning("Firebase client setup failed: %s", exc)
                self._ok = False

    @property
    def connected(self) -> bool:
        return self._ok and self._db is not None and self._bucket is not None

    # ── Version / stats ───────────────────────────────────────────────────────

    def get_current_version(self) -> str | None:
        """Return the latest index version tag from Firestore."""
        if not self.connected:
            return None
        try:
            doc = self._db.collection("index_versions").document("current").get()
            if doc.exists:
                return doc.to_dict().get("version")
        except Exception as exc:
            logger.warning("get_current_version failed: %s", exc)
        return None

    def get_artifact_counts(self) -> dict[str, int]:
        """Return {class: count} from Firestore index_versions/current."""
        if not self.connected:
            return {}
        try:
            doc = self._db.collection("index_versions").document("current").get()
            if doc.exists:
                return doc.to_dict().get("artifact_counts", {})
        except Exception as exc:
            logger.warning("get_artifact_counts failed: %s", exc)
        return {}

    # ── Embedding download ────────────────────────────────────────────────────

    def download_class_embeddings(
        self,
        class_name: str,
        version: str,
        on_progress: Callable | None = None,
    ) -> torch.Tensor | None:
        """
        Download pre-computed embedding tensor for a class from Storage.
        Returns [N, D] tensor or None if not found.
        """
        if not self.connected:
            return None
        path = f"embeddings/{version}/{class_name}.pt"
        try:
            if on_progress:
                on_progress(f"Downloading {class_name} embeddings from cloud...")
            blob = self._bucket.blob(path)
            if not blob.exists():
                return None
            buf = io.BytesIO()
            blob.download_to_file(buf)
            buf.seek(0)
            return torch.load(buf, map_location="cpu")
        except Exception as exc:
            logger.warning("download_class_embeddings(%s) failed: %s", class_name, exc)
            return None

    def download_class_metadata(
        self,
        class_name: str,
        version: str,
    ) -> list[dict]:
        """
        Download metadata JSON for a class from Storage.
        Returns list of {image_url, storage_path, source}.
        """
        if not self.connected:
            return []
        path = f"metadata/{version}/{class_name}.json"
        try:
            blob = self._bucket.blob(path)
            if not blob.exists():
                return []
            raw = blob.download_as_text()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("download_class_metadata(%s) failed: %s", class_name, exc)
            return []

    # ── Embedding upload ──────────────────────────────────────────────────────

    def upload_class_embeddings(
        self,
        class_name: str,
        embeddings: torch.Tensor,
        metadata: list[dict],
        version: str,
        on_progress: Callable | None = None,
    ) -> bool:
        """Upload embedding tensor + metadata for one class. Returns success."""
        if not self.connected:
            return False
        try:
            if on_progress:
                on_progress(f"Uploading {class_name} embeddings to cloud...")

            # Embeddings .pt
            buf = io.BytesIO()
            torch.save(embeddings, buf)
            buf.seek(0)
            self._bucket.blob(f"embeddings/{version}/{class_name}.pt").upload_from_file(
                buf, content_type="application/octet-stream"
            )

            # Metadata JSON
            meta_blob = self._bucket.blob(f"metadata/{version}/{class_name}.json")
            meta_blob.upload_from_string(
                json.dumps(metadata, ensure_ascii=False),
                content_type="application/json",
            )
            return True
        except Exception as exc:
            logger.warning("upload_class_embeddings(%s) failed: %s", class_name, exc)
            return False

    def update_version(self, version: str, artifact_counts: dict[str, int]) -> bool:
        """Update Firestore index_versions/current."""
        if not self.connected:
            return False
        try:
            from firebase_admin import firestore as fb_fs
            total = sum(artifact_counts.values())
            self._db.collection("index_versions").document("current").set({
                "version":         version,
                "updated_at":      fb_fs.SERVER_TIMESTAMP,
                "artifact_counts": artifact_counts,
                "total":           total,
            })
            return True
        except Exception as exc:
            logger.warning("update_version failed: %s", exc)
            return False

    # ── Image upload / download ───────────────────────────────────────────────

    def upload_image(
        self,
        local_path: Path,
        class_name: str,
        source: str = "met_museum",
        image_url: str = "",
    ) -> str | None:
        """
        Upload a local image to Firebase Storage.
        Returns the storage path on success, None on failure.
        """
        if not self.connected:
            return None
        storage_path = f"images/{class_name}/{local_path.name}"
        try:
            blob = self._bucket.blob(storage_path)
            if not blob.exists():   # skip already uploaded
                blob.upload_from_filename(str(local_path))
            return storage_path
        except Exception as exc:
            logger.warning("upload_image failed: %s", exc)
            return None

    def download_image(
        self,
        storage_path: str,
        dest: Path,
    ) -> bool:
        """Download an artifact image from Storage to local dest."""
        if not self.connected or dest.exists():
            return dest.exists()
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._bucket.blob(storage_path).download_to_filename(str(dest))
            return True
        except Exception as exc:
            logger.debug("download_image(%s) failed: %s", storage_path, exc)
            return False

    def get_image_signed_url(self, storage_path: str, expiry_minutes: int = 60) -> str | None:
        """Return a short-lived URL for an image (for display without downloading)."""
        if not self.connected:
            return None
        try:
            import datetime
            blob = self._bucket.blob(storage_path)
            return blob.generate_signed_url(
                expiration=datetime.timedelta(minutes=expiry_minutes),
                method="GET",
            )
        except Exception as exc:
            logger.debug("get_image_signed_url failed: %s", exc)
            return None

    # ── Correction sync ───────────────────────────────────────────────────────

    def upload_correction(
        self,
        predicted: str,
        correct: str,
        embedding: list[float],
    ) -> bool:
        """Push a user correction to Firestore corrections collection."""
        if not self.connected:
            return False
        try:
            from firebase_admin import firestore as fb_fs
            self._db.collection("corrections").add({
                "predicted":  predicted,
                "correct":    correct,
                "embedding":  embedding,
                "created_at": fb_fs.SERVER_TIMESTAMP,
            })
            return True
        except Exception as exc:
            logger.warning("upload_correction failed: %s", exc)
            return False

    def download_corrections(self, limit: int = 500) -> list[dict]:
        """Download recent corrections from Firestore for retraining."""
        if not self.connected:
            return []
        try:
            docs = (
                self._db.collection("corrections")
                .order_by("created_at", direction="DESCENDING")
                .limit(limit)
                .stream()
            )
            return [d.to_dict() for d in docs]
        except Exception as exc:
            logger.warning("download_corrections failed: %s", exc)
            return []


# ── Module-level helper ───────────────────────────────────────────────────────

_client: FirebaseClient | None = None


def get_client() -> FirebaseClient | None:
    """
    Return the singleton FirebaseClient, or None if not configured.
    Reads credentials from config.py.
    """
    global _client
    if _client is not None:
        return _client

    try:
        import config
        if not getattr(config, "FIREBASE_ENABLED", False):
            return None
        creds = getattr(config, "FIREBASE_CREDENTIALS", "")
        bucket = getattr(config, "FIREBASE_BUCKET", "")
        if not creds or not bucket:
            return None
        if not Path(creds).exists():
            logger.warning(
                "Firebase credentials file '%s' not found. Running offline.", creds
            )
            return None
        _client = FirebaseClient(creds, bucket)
        return _client if _client.connected else None
    except Exception as exc:
        logger.warning("get_client failed: %s", exc)
        return None
