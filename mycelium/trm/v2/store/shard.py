from __future__ import annotations
import json
import logging
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mycelium.trm.v2.store.schema import ALL_DDL

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pack(floats: List[float]) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


def _unpack(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class DomainShard:
    """
    SQLite shard for a single domain.

    Usage:
        shard = DomainShard.open("/data/shards", "mathematics")
        shard.insert_query(...)
        rows = shard.fetch_training_batch(limit=512)
        shard.close()

    One connection per shard — not thread-safe; use one shard per worker.
    """

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._apply_schema()

    @classmethod
    def open(cls, shard_root: str, domain_id: str) -> "DomainShard":
        p = Path(shard_root) / domain_id
        p.mkdir(parents=True, exist_ok=True)
        db_path = str(p / "queries.db")
        return cls(db_path)

    def _apply_schema(self) -> None:
        with self._conn:
            for ddl in ALL_DDL:
                self._conn.execute(ddl)

    # ------------------------------------------------------------------
    # Queries table
    # ------------------------------------------------------------------

    def insert_query(
        self,
        query_text: str,
        embedding: List[float],
        label: int = 1,
        confidence: float = 0.0,
        novelty_sim: float = 0.0,
        domain_version: int = 1,
        head_version: int = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        blob = _pack(embedding)
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO queries
                   (query_text, embedding_blob, dim, label, confidence, novelty_sim,
                    domain_version, head_version, routed_at, meta_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    query_text, blob, len(embedding), label,
                    confidence, novelty_sim,
                    domain_version, head_version,
                    _now(), json.dumps(meta or {}),
                ),
            )
        return cur.lastrowid

    def fetch_training_batch(
        self,
        limit: int = 1024,
        min_label: Optional[int] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns rows as dicts with 'embedding' already unpacked to List[float].
        """
        clauses = []
        params: List[Any] = []
        if min_label is not None:
            clauses.append("label >= ?")
            params.append(min_label)
        if since is not None:
            clauses.append("routed_at >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM queries {where} ORDER BY routed_at DESC LIMIT ?"
        params.append(limit)

        rows = []
        for row in self._conn.execute(sql, params):
            d = dict(row)
            d["embedding"] = _unpack(d.pop("embedding_blob"))
            rows.append(d)
        return rows

    def count_queries(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]

    def label_distribution(self) -> Dict[int, int]:
        rows = self._conn.execute(
            "SELECT label, COUNT(*) as cnt FROM queries GROUP BY label"
        ).fetchall()
        return {r["label"]: r["cnt"] for r in rows}

    # ------------------------------------------------------------------
    # Drift events table
    # ------------------------------------------------------------------

    def insert_drift_event(
        self,
        drift_type: str,
        magnitude: float,
        centroid_delta: float = 0.0,
        variance_ratio: float = 1.0,
        sample_count: int = 0,
        domain_version: int = 1,
        notes: str = "",
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO drift_events
                   (drift_type, magnitude, centroid_delta, variance_ratio,
                    sample_count, domain_version, detected_at, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (drift_type, magnitude, centroid_delta, variance_ratio,
                 sample_count, domain_version, _now(), notes),
            )
        return cur.lastrowid

    def unresolved_drift_events(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM drift_events WHERE resolved=0 ORDER BY detected_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def resolve_drift_event(self, event_id: int, notes: str = "") -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE drift_events SET resolved=1, notes=? WHERE id=?",
                (notes, event_id),
            )

    # ------------------------------------------------------------------
    # Labels table
    # ------------------------------------------------------------------

    def insert_label(
        self,
        query_id: int,
        label: int,
        source: str = "auto",
        notes: str = "",
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO labels (query_id, label, source, created_at, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (query_id, label, source, _now(), notes),
            )
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        self._conn.execute("VACUUM;")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DomainShard":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def path(self) -> str:
        return self._path
