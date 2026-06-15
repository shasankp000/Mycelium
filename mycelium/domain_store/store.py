from __future__ import annotations
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mycelium.domain_store.schema import ALL_DDL

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteDomainStore:
    """
    One SQLite file per domain.
    Path: {root_dir}/{domain_id}.sqlite
    """

    def __init__(self, root_dir: str, domain_id: str) -> None:
        self.domain_id = domain_id
        self._path = Path(root_dir) / f"{domain_id}.sqlite"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._apply_schema()
        logger.info("SQLiteDomainStore opened: %s", self._path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _apply_schema(self) -> None:
        cur = self._conn.cursor()
        for stmt in ALL_DDL:
            try:
                cur.executescript(stmt)
            except sqlite3.OperationalError as e:
                logger.warning("DDL warning (%s): %s", stmt[:40], e)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def insert_chunk(
        self,
        text: str,
        source: Optional[str] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        chunk_id: Optional[str] = None,
    ) -> str:
        cid = chunk_id or str(uuid.uuid4())
        self._conn.execute(
            "INSERT OR IGNORE INTO chunks (chunk_id, domain_id, source, text, char_start, char_end, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cid, self.domain_id, source, text, char_start, char_end, _now()),
        )
        self._conn.commit()
        return cid

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def chunk_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def upsert_embedding(
        self, chunk_id: str, vector_bytes: bytes, dim: int, model_tag: str
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (chunk_id, vector, dim, model_tag) VALUES (?, ?, ?, ?)",
            (chunk_id, vector_bytes, dim, model_tag),
        )
        self._conn.commit()

    def get_embedding(self, chunk_id: str) -> Optional[Tuple[bytes, int, str]]:
        row = self._conn.execute(
            "SELECT vector, dim, model_tag FROM embeddings WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return (row["vector"], row["dim"], row["model_tag"]) if row else None

    def all_embeddings(self) -> List[Tuple[str, bytes, int]]:
        """Returns [(chunk_id, vector_bytes, dim), ...]"""
        rows = self._conn.execute(
            "SELECT chunk_id, vector, dim FROM embeddings"
        ).fetchall()
        return [(r["chunk_id"], r["vector"], r["dim"]) for r in rows]

    # ------------------------------------------------------------------
    # FTS5 search
    # ------------------------------------------------------------------

    def fts_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT c.chunk_id, c.text, c.source, bm25(chunks_fts) AS score "
            "FROM chunks_fts "
            "JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? "
            "ORDER BY score LIMIT ?",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Shard metadata
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO shard_meta (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM shard_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    # ------------------------------------------------------------------
    # Replay samples
    # ------------------------------------------------------------------

    def insert_replay_sample(
        self,
        query_text: str,
        label: Optional[str] = None,
        score: Optional[float] = None,
    ) -> str:
        sid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO replay_samples (sample_id, domain_id, query_text, label, score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, self.domain_id, query_text, label, score, _now()),
        )
        self._conn.commit()
        return sid

    def replay_sample_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM replay_samples").fetchone()[0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteDomainStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
