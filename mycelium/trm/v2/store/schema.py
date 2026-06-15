"""
DDL for per-domain SQLite shards.

One shard per domain_id, stored at:
  <shard_root>/<domain_id>/queries.db

Tables:
  queries       — every routed query embedding + metadata
  drift_events  — drift signals recorded for this domain
  labels        — manual / auto label corrections for replay training
"""

QUERIES_TABLE = """
CREATE TABLE IF NOT EXISTS queries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text      TEXT    NOT NULL,
    embedding_blob  BLOB    NOT NULL,   -- packed float32 array
    dim             INTEGER NOT NULL,
    label           INTEGER NOT NULL DEFAULT 1,  -- 1=in-domain, 0=negative
    confidence      REAL    NOT NULL DEFAULT 0.0,
    novelty_sim     REAL    NOT NULL DEFAULT 0.0,
    domain_version  INTEGER NOT NULL DEFAULT 1,
    head_version    INTEGER NOT NULL DEFAULT 0,
    routed_at       TEXT    NOT NULL,
    meta_json       TEXT    NOT NULL DEFAULT '{}'
);
"""

DRIFT_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS drift_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    drift_type      TEXT    NOT NULL,
    magnitude       REAL    NOT NULL,
    centroid_delta  REAL    NOT NULL DEFAULT 0.0,
    variance_ratio  REAL    NOT NULL DEFAULT 1.0,
    sample_count    INTEGER NOT NULL DEFAULT 0,
    domain_version  INTEGER NOT NULL DEFAULT 1,
    detected_at     TEXT    NOT NULL,
    resolved        INTEGER NOT NULL DEFAULT 0,
    notes           TEXT    NOT NULL DEFAULT ''
);
"""

LABELS_TABLE = """
CREATE TABLE IF NOT EXISTS labels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id        INTEGER NOT NULL REFERENCES queries(id),
    label           INTEGER NOT NULL,   -- corrected label
    source          TEXT    NOT NULL DEFAULT 'auto',  -- 'auto' | 'human'
    created_at      TEXT    NOT NULL,
    notes           TEXT    NOT NULL DEFAULT ''
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_queries_routed_at   ON queries(routed_at);",
    "CREATE INDEX IF NOT EXISTS idx_queries_label        ON queries(label);",
    "CREATE INDEX IF NOT EXISTS idx_drift_detected_at   ON drift_events(detected_at);",
    "CREATE INDEX IF NOT EXISTS idx_labels_query_id     ON labels(query_id);",
]

ALL_DDL = [QUERIES_TABLE, DRIFT_EVENTS_TABLE, LABELS_TABLE] + INDEXES
