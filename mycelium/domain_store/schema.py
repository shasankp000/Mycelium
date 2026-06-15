"""
SQL DDL for per-domain SQLite shards.
Each domain gets its own .sqlite file at artifacts/sqlite_experts/{domain_id}.sqlite
"""

CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    domain_id   TEXT NOT NULL,
    source      TEXT,
    text        TEXT NOT NULL,
    char_start  INTEGER,
    char_end    INTEGER,
    created_at  TEXT NOT NULL
);
"""

CREATE_CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(chunk_id UNINDEXED, text, content='chunks', content_rowid='rowid');
"""

CREATE_CHUNKS_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (new.rowid, new.chunk_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text) VALUES('delete', old.rowid, old.chunk_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text) VALUES('delete', old.rowid, old.chunk_id, old.text);
  INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (new.rowid, new.chunk_id, new.text);
END;
"""

CREATE_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id  TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    vector    BLOB NOT NULL,
    dim       INTEGER NOT NULL,
    model_tag TEXT NOT NULL
);
"""

CREATE_ENTITIES = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id   TEXT PRIMARY KEY,
    chunk_id    TEXT REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    entity_type TEXT,
    value       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

CREATE_RELATIONS = """
CREATE TABLE IF NOT EXISTS relations (
    relation_id  TEXT PRIMARY KEY,
    source_id    TEXT REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_id    TEXT REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation     TEXT NOT NULL,
    confidence   REAL DEFAULT 1.0,
    created_at   TEXT NOT NULL
);
"""

CREATE_SHARD_META = """
CREATE TABLE IF NOT EXISTS shard_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""

CREATE_REPLAY_SAMPLES = """
CREATE TABLE IF NOT EXISTS replay_samples (
    sample_id   TEXT PRIMARY KEY,
    domain_id   TEXT NOT NULL,
    query_text  TEXT NOT NULL,
    label       TEXT,
    score       REAL,
    created_at  TEXT NOT NULL
);
"""

ALL_DDL = [
    CREATE_CHUNKS,
    CREATE_CHUNKS_FTS,
    CREATE_CHUNKS_FTS_TRIGGERS,
    CREATE_EMBEDDINGS,
    CREATE_ENTITIES,
    CREATE_RELATIONS,
    CREATE_SHARD_META,
    CREATE_REPLAY_SAMPLES,
]
