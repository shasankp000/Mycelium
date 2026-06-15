from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from mycelium.trm.v2.store.shard import DomainShard

logger = logging.getLogger(__name__)


class QueryStore:
    """
    Manages per-domain DomainShard instances.

    Shards are opened lazily on first access and kept open for the
    lifetime of this object. Call close_all() at shutdown.

    Contract:
      - One shard per domain_id, one file per domain.
      - All writes go through record_query() — no direct shard access in
        production code.
      - Shards are independent; a crash in one domain does not affect others.
    """

    def __init__(self, shard_root: str) -> None:
        self._root = shard_root
        self._shards: Dict[str, DomainShard] = {}

    # ------------------------------------------------------------------
    # Shard access
    # ------------------------------------------------------------------

    def shard(self, domain_id: str) -> DomainShard:
        if domain_id not in self._shards:
            self._shards[domain_id] = DomainShard.open(self._root, domain_id)
            logger.debug("QueryStore: opened shard for domain '%s'.", domain_id)
        return self._shards[domain_id]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_query(
        self,
        domain_id: str,
        query_text: str,
        embedding: List[float],
        label: int = 1,
        confidence: float = 0.0,
        novelty_sim: float = 0.0,
        domain_version: int = 1,
        head_version: int = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        return self.shard(domain_id).insert_query(
            query_text=query_text,
            embedding=embedding,
            label=label,
            confidence=confidence,
            novelty_sim=novelty_sim,
            domain_version=domain_version,
            head_version=head_version,
            meta=meta,
        )

    def record_drift(
        self,
        domain_id: str,
        drift_type: str,
        magnitude: float,
        centroid_delta: float = 0.0,
        variance_ratio: float = 1.0,
        sample_count: int = 0,
        domain_version: int = 1,
        notes: str = "",
    ) -> int:
        return self.shard(domain_id).insert_drift_event(
            drift_type=drift_type,
            magnitude=magnitude,
            centroid_delta=centroid_delta,
            variance_ratio=variance_ratio,
            sample_count=sample_count,
            domain_version=domain_version,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def fetch_training_batch(
        self,
        domain_id: str,
        limit: int = 1024,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.shard(domain_id).fetch_training_batch(limit=limit, since=since)

    def stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            did: {
                "query_count": s.count_queries(),
                "label_distribution": s.label_distribution(),
                "path": s.path,
            }
            for did, s in self._shards.items()
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close_all(self) -> None:
        for did, shard in self._shards.items():
            shard.close()
            logger.debug("QueryStore: closed shard '%s'.", did)
        self._shards.clear()

    def __enter__(self) -> "QueryStore":
        return self

    def __exit__(self, *_) -> None:
        self.close_all()
