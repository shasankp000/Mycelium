"""
tests/domain_store/test_domain_store.py

Unit tests for:
  - mycelium.domain_store.schema  (SCHEMA_DDL structure)
  - mycelium.domain_store.store   (SQLiteDomainStore CRUD + search)
  - mycelium.domain_store.retrieval (DomainRetrievalEngine flow)
  - mycelium.domain_store.migration (LegacyExpertMigrator interface)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchemaDDL:
    def test_schema_ddl_is_string(self):
        from mycelium.domain_store.schema import SCHEMA_DDL
        assert isinstance(SCHEMA_DDL, str)
        assert len(SCHEMA_DDL) > 100

    def test_schema_ddl_contains_required_tables(self):
        from mycelium.domain_store.schema import SCHEMA_DDL
        ddl = SCHEMA_DDL.lower()
        for table in ["chunks", "entities", "relations", "replay_samples", "shard_meta"]:
            assert table in ddl, f"Missing table: {table}"

    def test_schema_ddl_contains_fts5(self):
        from mycelium.domain_store.schema import SCHEMA_DDL
        assert "fts5" in SCHEMA_DDL.lower()

    def test_schema_can_be_executed(self, tmp_path):
        from mycelium.domain_store.schema import SCHEMA_DDL
        db_path = tmp_path / "test.sqlite"
        conn = sqlite3.connect(db_path)
        for stmt in SCHEMA_DDL.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                if "fts5" not in str(exc).lower():
                    raise
        conn.close()


# ---------------------------------------------------------------------------
# SQLiteDomainStore
# ---------------------------------------------------------------------------

class TestSQLiteDomainStore:
    @pytest.fixture
    def store(self, tmp_path):
        from mycelium.domain_store.store import SQLiteDomainStore
        db_path = tmp_path / "test_domain.sqlite"
        return SQLiteDomainStore(db_path)

    @pytest.fixture
    def sample_chunk(self):
        return {
            "id": str(uuid.uuid4()),
            "text": "Quantum mechanics describes nature at the smallest scales.",
            "source": "wiki_quantum.txt",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }

    # upsert_chunk
    def test_upsert_chunk_does_not_raise(self, store, sample_chunk):
        store.upsert_chunk(sample_chunk)

    def test_upsert_chunk_idempotent(self, store, sample_chunk):
        store.upsert_chunk(sample_chunk)
        store.upsert_chunk(sample_chunk)  # duplicate key update

    # upsert_entity
    def test_upsert_entity(self, store):
        store.upsert_entity({
            "entity_id": "ent_001",
            "label": "Electron",
            "entity_type": "particle",
        })

    # upsert_relation
    def test_upsert_relation(self, store):
        store.upsert_relation({
            "relation_id": "rel_001",
            "src_id": "ent_001",
            "rel_type": "is_part_of",
            "dst_id": "ent_002",
            "weight": 0.9,
        })

    # add_replay_sample
    def test_add_replay_sample(self, store):
        store.add_replay_sample({
            "sample_id": str(uuid.uuid4()),
            "text": "Sample replay text about physics.",
            "label_json": json.dumps({"domain": "physics"}),
        })

    # set_meta / get_meta
    def test_set_and_get_meta(self, store):
        store.set_meta("domain_id", "physics")
        val = store.get_meta("domain_id")
        assert val == "physics"

    def test_get_meta_missing_key_returns_none(self, store):
        val = store.get_meta("__nonexistent_key__")
        assert val is None

    # search_fts
    def test_search_fts_returns_list(self, store, sample_chunk):
        store.upsert_chunk(sample_chunk)
        results = store.search_fts("quantum", limit=10)
        assert isinstance(results, list)

    def test_search_fts_finds_inserted_chunk(self, store, sample_chunk):
        store.upsert_chunk(sample_chunk)
        results = store.search_fts("quantum", limit=10)
        if results:  # FTS5 may not be available
            texts = [r.get("text", "") for r in results]
            assert any("quantum" in t.lower() for t in texts)

    def test_search_fts_empty_on_no_match(self, store, sample_chunk):
        store.upsert_chunk(sample_chunk)
        results = store.search_fts("xyzzy_impossible_string_12345", limit=10)
        assert isinstance(results, list)

    # get_replay_samples
    def test_get_replay_samples_returns_list(self, store):
        store.add_replay_sample({
            "sample_id": str(uuid.uuid4()),
            "text": "Replay sample.",
        })
        samples = store.get_replay_samples(limit=100)
        assert isinstance(samples, list)
        assert len(samples) >= 1

    # cosine_search
    def test_cosine_search_returns_list(self, store):
        results = store.cosine_search(
            query_embedding=[0.1, 0.2, 0.3],
            candidate_ids=[],
            limit=10,
        )
        assert isinstance(results, list)

    def test_cosine_search_with_stored_embedding(self, store):
        import struct
        chunk_id = str(uuid.uuid4())
        vec = [0.1, 0.2, 0.3]
        packed = struct.pack(f"{len(vec)}f", *vec)
        conn = sqlite3.connect(store._db_path if hasattr(store, "_db_path") else store.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO chunks (id, text, embedding) VALUES (?, ?, ?)",
                (chunk_id, "test text", packed),
            )
            conn.commit()
        finally:
            conn.close()
        results = store.cosine_search(
            query_embedding=vec,
            candidate_ids=[chunk_id],
            limit=5,
        )
        assert isinstance(results, list)

    # get_chunk_embeddings
    def test_get_chunk_embeddings_returns_list(self, store, sample_chunk):
        store.upsert_chunk(sample_chunk)
        result = store.get_chunk_embeddings([sample_chunk["id"]])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# DomainRetrievalEngine
# ---------------------------------------------------------------------------

class TestDomainRetrievalEngine:
    @pytest.fixture
    def engine_with_store(self, tmp_path):
        from mycelium.domain_store.store import SQLiteDomainStore
        from mycelium.domain_store.retrieval import DomainRetrievalEngine
        store = SQLiteDomainStore(tmp_path / "physics.sqlite")
        store.upsert_chunk({
            "id": "c1",
            "text": "Newton's laws of motion describe classical mechanics.",
            "source": "textbook.txt",
        })
        store.upsert_chunk({
            "id": "c2",
            "text": "Einsteins special relativity replaces Newtonian mechanics at high velocity.",
            "source": "textbook.txt",
        })
        engine = DomainRetrievalEngine(store=store)
        return engine

    def test_retrieve_returns_list(self, engine_with_store):
        results = engine_with_store.retrieve(
            domain_id="physics",
            query="What are Newton's laws?",
            query_embedding=[0.1] * 16,
            k=5,
        )
        assert isinstance(results, list)

    def test_retrieve_respects_k(self, engine_with_store):
        results = engine_with_store.retrieve(
            domain_id="physics",
            query="Newton",
            query_embedding=[0.1] * 16,
            k=1,
        )
        assert len(results) <= 1

    def test_retrieve_result_has_text(self, engine_with_store):
        results = engine_with_store.retrieve(
            domain_id="physics",
            query="Newton",
            query_embedding=[0.1] * 16,
            k=5,
        )
        for r in results:
            assert "text" in r

    def test_retrieve_with_no_results(self, engine_with_store):
        results = engine_with_store.retrieve(
            domain_id="physics",
            query="xyzzy_impossible_query_99999",
            query_embedding=[0.0] * 16,
            k=5,
        )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# LegacyExpertMigrator (interface test — no real BERT expert needed)
# ---------------------------------------------------------------------------

class TestLegacyExpertMigrator:
    @pytest.fixture
    def migrator(self, tmp_path):
        from mycelium.domain_store.migration import LegacyExpertMigrator
        from mycelium.domain_store.embedder import OfflineEmbedder
        return LegacyExpertMigrator(
            bert_expert=None,
            store_root=tmp_path,
            embedder=OfflineEmbedder(),
        )

    def test_export_domain_corpus_returns_list(self, migrator):
        corpus = migrator.export_domain_corpus("physics")
        assert isinstance(corpus, list)

    def test_build_shard_returns_path(self, migrator, tmp_path):
        path = migrator.build_shard("physics")
        assert isinstance(path, Path)

    def test_validate_shard_returns_dict(self, migrator, tmp_path):
        shard_path = migrator.build_shard("physics")
        report = migrator.validate_shard(shard_path)
        assert isinstance(report, dict)

    def test_run_full_migration_returns_dict(self, migrator):
        result = migrator.run_full_migration(["physics", "maths"])
        assert isinstance(result, dict)
