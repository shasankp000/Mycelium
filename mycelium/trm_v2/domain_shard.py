"""
mycelium/trm_v2/domain_shard.py

Per-domain SQLite knowledge store — replaces BERT-based domain expert models.

Schema (per shard):
    chunks      – raw text + embedding blob + source provenance
    fts_chunks  – FTS5 virtual table for lexical prefiltering
    entities    – named entities with type + metadata
    relations   – typed directed edges between entities
    artifacts   – arbitrary key/value domain metadata
    replay_samples – labelled samples for cold-storage replay

Retrieval policy (QueryStore):
    Step 1 — FTS5 lexical prefilter  → candidate set
    Step 2 — cosine vector similarity → ranked candidates
    Step 3 — graph/metadata reranking (entity/relation boost)
    Step 4 — top-k handoff to reasoning pipeline

Spec refs:
    mycelium_trm_v2_theoretical_spec_v0.2.md  §SQLite-Sharded Expert Theory
    mycelium_understanding.md                  §Phase 1 scope
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    embedding     BLOB,          -- float32 array, little-endian
    source        TEXT,
    created_at    REAL NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks
    USING fts5(text, content='chunks', content_rowid='rowid');

-- Keep FTS in sync with chunks via triggers
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts_chunks(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, text)
        VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, text)
        VALUES ('delete', old.rowid, old.text);
    INSERT INTO fts_chunks(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS entities (
    entity_id     TEXT PRIMARY KEY,
    label         TEXT NOT NULL,
    type          TEXT,
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS relations (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id   TEXT NOT NULL REFERENCES entities(entity_id),
    rel_type TEXT NOT NULL,
    dst_id   TEXT NOT NULL REFERENCES entities(entity_id),
    weight   REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS artifacts (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS replay_samples (
    sample_id     TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    label_json    TEXT DEFAULT '{}',
    embedding     BLOB,
    metadata_json TEXT DEFAULT '{}',
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source   ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_relations_src   ON relations(src_id);
CREATE INDEX IF NOT EXISTS idx_relations_dst   ON relations(dst_id);
"""


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _encode_embedding(vec: np.ndarray) -> bytes:
    """Pack float32 ndarray to little-endian bytes."""
    return struct.pack(f"<{len(vec)}f", *vec.astype(np.float32))


def _decode_embedding(blob: bytes) -> np.ndarray:
    """Unpack little-endian bytes to float32 ndarray."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    chunk_id:  str
    text:      str
    score:     float          # final reranked score
    source:    Optional[str]
    metadata:  Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DomainShard — one SQLite file per domain
# ---------------------------------------------------------------------------

class DomainShard:
    """SQLite-backed knowledge store for a single domain.

    Lifecycle:
        shard = DomainShard("/path/to/domain.db")
        shard.open()          # creates schema if new
        shard.insert_chunk(text, embedding, source)
        shard.close()

    The shard is also the natural cold-storage persistence unit:
    when a domain is offloaded (COLD state) the file stays on disk
    and is reopened on reactivation (REMEMBERING -> HOT).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ---- Connection management ---------------------------------------------

    def open(self) -> None:
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()
        logger.debug("DomainShard opened: %s", self.db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("DomainShard closed: %s", self.db_path)

    def __enter__(self) -> "DomainShard":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield the connection inside a transaction; rollback on error."""
        if self._conn is None:
            raise RuntimeError("DomainShard is not open. Call open() first.")
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _apply_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ---- Chunk operations --------------------------------------------------

    def insert_chunk(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id: Optional[str] = None,
    ) -> str:
        """Insert a text chunk. Returns the assigned chunk_id."""
        cid = chunk_id or str(uuid.uuid4())
        emb_blob = _encode_embedding(embedding) if embedding is not None else None
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO chunks (id, text, embedding, source, created_at, metadata_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    text,
                    emb_blob,
                    source,
                    time.time(),
                    json.dumps(metadata or {}),
                ),
            )
        return cid

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError("DomainShard is not open.")
        row = self._conn.execute(
            "SELECT id, text, source, created_at, metadata_json FROM chunks WHERE id=?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def chunk_count(self) -> int:
        if self._conn is None:
            raise RuntimeError("DomainShard is not open.")
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ---- Entity / relation operations --------------------------------------

    def insert_entity(
        self,
        label: str,
        entity_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
    ) -> str:
        eid = entity_id or str(uuid.uuid4())
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO entities (entity_id, label, type, metadata_json)"
                " VALUES (?, ?, ?, ?)",
                (eid, label, entity_type, json.dumps(metadata or {})),
            )
        return eid

    def insert_relation(
        self,
        src_id: str,
        rel_type: str,
        dst_id: str,
        weight: float = 1.0,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO relations (src_id, rel_type, dst_id, weight)"
                " VALUES (?, ?, ?, ?)",
                (src_id, rel_type, dst_id, weight),
            )

    # ---- Artifact operations -----------------------------------------------

    def set_artifact(self, key: str, value: Any) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    def get_artifact(self, key: str) -> Optional[Any]:
        if self._conn is None:
            raise RuntimeError("DomainShard is not open.")
        row = self._conn.execute(
            "SELECT value FROM artifacts WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    # ---- Replay sample operations ------------------------------------------

    def insert_replay_sample(
        self,
        text: str,
        label: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sample_id: Optional[str] = None,
    ) -> str:
        sid = sample_id or str(uuid.uuid4())
        emb_blob = _encode_embedding(embedding) if embedding is not None else None
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO replay_samples"
                " (sample_id, text, label_json, embedding, metadata_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    text,
                    json.dumps(label or {}),
                    emb_blob,
                    json.dumps(metadata or {}),
                    time.time(),
                ),
            )
        return sid

    def get_replay_samples(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Return up to `limit` replay samples ordered by recency."""
        if self._conn is None:
            raise RuntimeError("DomainShard is not open.")
        rows = self._conn.execute(
            "SELECT sample_id, text, label_json, metadata_json, created_at"
            " FROM replay_samples ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def replay_sample_count(self) -> int:
        if self._conn is None:
            raise RuntimeError("DomainShard is not open.")
        return self._conn.execute("SELECT COUNT(*) FROM replay_samples").fetchone()[0]


# ---------------------------------------------------------------------------
# QueryStore — layered retrieval over a DomainShard
# ---------------------------------------------------------------------------

class QueryStore:
    """Implements the four-step retrieval policy over a DomainShard.

    Usage:
        qs = QueryStore(shard)
        results = qs.retrieve(query_embedding, query_text, top_k=5)
    """

    def __init__(
        self,
        shard: DomainShard,
        fts_candidate_limit: int = 50,
        entity_boost: float = 0.05,
    ) -> None:
        self.shard = shard
        self.fts_candidate_limit = fts_candidate_limit
        self.entity_boost = entity_boost

    # ---- Step 1: FTS5 lexical prefilter ------------------------------------

    def _fts_candidates(self, query_text: str) -> List[str]:
        """Return chunk IDs that match query_text via FTS5."""
        if self.shard._conn is None:
            raise RuntimeError("DomainShard is not open.")
        # Escape FTS5 special chars to avoid query syntax errors
        safe = query_text.replace('"', '""')
        try:
            rows = self.shard._conn.execute(
                "SELECT c.id FROM fts_chunks f"
                " JOIN chunks c ON c.rowid = f.rowid"
                " WHERE fts_chunks MATCH ?"
                " LIMIT ?",
                (f'"{safe}"', self.fts_candidate_limit),
            ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.OperationalError:
            # FTS query parse error — fall back to full scan
            logger.warning("FTS query failed for '%s', falling back to full scan.", query_text)
            rows = self.shard._conn.execute(
                "SELECT id FROM chunks LIMIT ?", (self.fts_candidate_limit,)
            ).fetchall()
            return [r[0] for r in rows]

    # ---- Step 2: vector similarity -----------------------------------------

    def _vector_rank(
        self,
        candidate_ids: List[str],
        query_embedding: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """Return (chunk_id, cosine_score) sorted descending."""
        if self.shard._conn is None:
            raise RuntimeError("DomainShard is not open.")
        if not candidate_ids:
            return []
        placeholders = ",".join("?" * len(candidate_ids))
        rows = self.shard._conn.execute(
            f"SELECT id, embedding FROM chunks WHERE id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
        scored: List[Tuple[str, float]] = []
        for row in rows:
            if row["embedding"] is None:
                scored.append((row["id"], 0.0))
                continue
            vec = _decode_embedding(row["embedding"])
            scored.append((row["id"], _cosine(query_embedding, vec)))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    # ---- Step 3: graph/entity reranking ------------------------------------

    def _entity_boost_ids(self, query_text: str) -> set:
        """Return chunk IDs whose source is linked to a query-matching entity."""
        if self.shard._conn is None:
            return set()
        safe = query_text.replace('"', '""')
        try:
            rows = self.shard._conn.execute(
                "SELECT c.id FROM chunks c"
                " JOIN entities e ON c.source = e.label"
                " WHERE lower(e.label) LIKE lower(?)"
                " LIMIT 20",
                (f"%{query_text}%",),
            ).fetchall()
            return {r[0] for r in rows}
        except Exception:
            return set()

    # ---- Step 4: top-k handoff ---------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
        top_k: int = 5,
    ) -> List[ChunkResult]:
        """Full four-step retrieval. Returns up to top_k ChunkResult objects.

        If query_text is empty, FTS is skipped and all chunks are considered
        for vector ranking (up to fts_candidate_limit).
        """
        if self.shard._conn is None:
            raise RuntimeError("DomainShard is not open.")

        # Step 1 — FTS prefilter
        if query_text.strip():
            candidate_ids = self._fts_candidates(query_text)
        else:
            rows = self.shard._conn.execute(
                "SELECT id FROM chunks LIMIT ?", (self.fts_candidate_limit,)
            ).fetchall()
            candidate_ids = [r[0] for r in rows]

        if not candidate_ids:
            return []

        # Step 2 — vector rank
        ranked = self._vector_rank(candidate_ids, query_embedding)

        # Step 3 — entity boost
        boosted_ids = self._entity_boost_ids(query_text) if query_text.strip() else set()
        boosted = [
            (cid, score + self.entity_boost if cid in boosted_ids else score)
            for cid, score in ranked
        ]
        boosted.sort(key=lambda x: x[1], reverse=True)

        # Step 4 — fetch top-k metadata and return
        results: List[ChunkResult] = []
        for cid, score in boosted[:top_k]:
            row = self.shard._conn.execute(
                "SELECT text, source, metadata_json FROM chunks WHERE id=?", (cid,)
            ).fetchone()
            if row is None:
                continue
            results.append(
                ChunkResult(
                    chunk_id=cid,
                    text=row["text"],
                    score=score,
                    source=row["source"],
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
            )
        return results
