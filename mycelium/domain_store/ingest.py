from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from mycelium.domain_store.store import SQLiteDomainStore
from mycelium.domain_store.embedder import OfflineEmbedder

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 512      # characters
DEFAULT_CHUNK_OVERLAP = 64    # characters


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[tuple[str, int, int]]:
    """Yields (chunk_text, char_start, char_end)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((text[start:end], start, end))
        start += chunk_size - overlap
    return chunks


class DomainIngestor:
    """
    Ingests raw text into a SQLiteDomainStore shard.
    Called offline (corpus loading, migration) — never at inference time.
    """

    def __init__(
        self,
        store: SQLiteDomainStore,
        embedder: Optional[OfflineEmbedder] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._overlap = overlap

    def ingest_text(
        self,
        text: str,
        source: Optional[str] = None,
    ) -> List[str]:
        """
        Chunk `text`, store chunks, optionally embed.
        Returns list of chunk_ids inserted.
        """
        chunks = _chunk_text(text, self._chunk_size, self._overlap)
        chunk_ids = []
        texts_to_embed = []

        for chunk_text, start, end in chunks:
            cid = self._store.insert_chunk(
                text=chunk_text,
                source=source,
                char_start=start,
                char_end=end,
            )
            chunk_ids.append(cid)
            texts_to_embed.append(chunk_text)

        if self._embedder and texts_to_embed:
            blobs = self._embedder.embed(texts_to_embed)
            dim = self._embedder.dim
            tag = self._embedder.model_name
            for cid, blob in zip(chunk_ids, blobs):
                self._store.upsert_embedding(cid, blob, dim, tag)

        logger.info(
            "Ingested %d chunks into domain '%s' from source '%s'",
            len(chunk_ids), self._store.domain_id, source or "<inline>"
        )
        return chunk_ids

    def ingest_file(self, path: str | Path) -> List[str]:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(text, source=str(p))

    def ingest_files(self, paths: Iterable[str | Path]) -> int:
        total = 0
        for p in paths:
            ids = self.ingest_file(p)
            total += len(ids)
        return total
