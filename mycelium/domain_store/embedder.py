from __future__ import annotations
import logging
import struct
from typing import List

logger = logging.getLogger(__name__)


def _pack_floats(values: List[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _unpack_floats(data: bytes) -> List[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


class OfflineEmbedder:
    """
    Wraps a sentence-transformer model for offline ingest-time embedding.
    NEVER instantiated on the inference path — only called from DomainIngestor
    and migration scripts.

    model_name: any sentence-transformers compatible model name or local path.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None  # lazy load

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("OfflineEmbedder loaded model: %s", self.model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for OfflineEmbedder. "
                    "Install with: pip install sentence-transformers"
                ) from e

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[bytes]:
        """Returns a list of float32 byte blobs, one per text."""
        self._ensure_loaded()
        vecs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [_pack_floats(v.tolist()) for v in vecs]

    def embed_one(self, text: str) -> bytes:
        return self.embed([text])[0]

    @staticmethod
    def unpack(blob: bytes) -> List[float]:
        return _unpack_floats(blob)
