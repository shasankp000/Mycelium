from __future__ import annotations
import logging
import math
import struct
from typing import Dict, List, Optional, Tuple

from mycelium.domain_store.store import SQLiteDomainStore

logger = logging.getLogger(__name__)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _unpack(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class DomainRetrievalEngine:
    """
    Two-stage retrieval for a single domain shard:
      1. FTS5 BM25 prefilter (fast, lexical)
      2. Python cosine rerank over retrieved chunk embeddings

    query_vector: pre-computed embedding of the query (from SharedEncoder at inference time).
                  If None, returns FTS5 results as-is without reranking.
    """

    def __init__(self, store: SQLiteDomainStore, fts_limit: int = 20) -> None:
        self._store = store
        self._fts_limit = fts_limit

    def retrieve(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Returns up to `top_k` results sorted by relevance.
        Each result: {chunk_id, text, source, score, stage}
        """
        candidates = self._store.fts_search(query_text, limit=self._fts_limit)

        if not candidates:
            return []

        if query_vector is None:
            for r in candidates:
                r["stage"] = "fts"
            return candidates[:top_k]

        reranked = []
        for c in candidates:
            emb_row = self._store.get_embedding(c["chunk_id"])
            if emb_row is None:
                c["score"] = float(c.get("score", 0.0))
                c["stage"] = "fts_no_emb"
                reranked.append(c)
                continue
            blob, _, _ = emb_row
            vec = _unpack(blob)
            sim = _cosine(query_vector, vec)
            c["score"] = sim
            c["stage"] = "cosine"
            reranked.append(c)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        logger.debug(
            "Retrieval for domain '%s': %d fts candidates → top %d after rerank",
            self._store.domain_id, len(candidates), top_k,
        )
        return reranked[:top_k]
