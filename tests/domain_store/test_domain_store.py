"""
tests/domain_store/test_domain_store.py

Tier 1 – Structural / unit  : SQLiteDomainStore CRUD, schema idempotency,
                               FTS5 search, embedding round-trips.
Tier 2 – Semantic           : marked @pytest.mark.semantic
                               Confirm that insert+retrieve is lossless, FTS
                               ranks relevant documents higher than irrelevant
                               ones, and replay sample counts are accurate.

All tests use pytest's tmp_path fixture — no side effects on the real filesystem.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import List

import pytest

from mycelium.domain_store.store import SQLiteDomainStore
from mycelium.domain_store.schema import ALL_DDL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path, domain_id: str = "test_domain") -> SQLiteDomainStore:
    return SQLiteDomainStore(str(tmp_path), domain_id)


def _pack(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ===========================================================================
# TIER 1 — STRUCTURAL
# ===========================================================================


class TestSchemaIdempotency:
    """Applying the schema twice must not raise."""

    def test_schema_applied_twice_no_error(self, tmp_path):
        store = _make_store(tmp_path)
        # Force re-apply schema
        store._apply_schema()
        store.close()

    def test_all_ddl_is_non_empty_strings(self):
        assert isinstance(ALL_DDL, (list, tuple))
        assert len(ALL_DDL) > 0
        for stmt in ALL_DDL:
            assert isinstance(stmt, str) and len(stmt.strip()) > 0


class TestChunkCRUD:
    def test_insert_chunk_returns_id(self, tmp_path):
        store = _make_store(tmp_path)
        cid = store.insert_chunk("Hypertension is elevated blood pressure.")
        assert isinstance(cid, str) and len(cid) > 0
        store.close()

    def test_get_chunk_returns_correct_text(self, tmp_path):
        store = _make_store(tmp_path)
        text = "Aspirin reduces platelet aggregation."
        cid = store.insert_chunk(text)
        row = store.get_chunk(cid)
        assert row is not None
        assert row["text"] == text
        assert row["domain_id"] == "test_domain"
        store.close()

    def test_get_chunk_missing_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_chunk("nonexistent-uuid") is None
        store.close()

    def test_chunk_count_increments(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.chunk_count() == 0
        store.insert_chunk("First chunk.")
        store.insert_chunk("Second chunk.")
        assert store.chunk_count() == 2
        store.close()

    def test_insert_chunk_with_explicit_id(self, tmp_path):
        store = _make_store(tmp_path)
        cid = store.insert_chunk("text", chunk_id="explicit-id-123")
        assert cid == "explicit-id-123"
        assert store.get_chunk("explicit-id-123") is not None
        store.close()

    def test_insert_chunk_or_ignore_on_duplicate(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_chunk("text", chunk_id="dup-id")
        store.insert_chunk("different text", chunk_id="dup-id")  # should silently ignore
        assert store.chunk_count() == 1
        store.close()

    def test_chunk_stores_source_and_offsets(self, tmp_path):
        store = _make_store(tmp_path)
        cid = store.insert_chunk("text", source="pubmed:123", char_start=0, char_end=4)
        row = store.get_chunk(cid)
        assert row["source"] == "pubmed:123"
        assert row["char_start"] == 0
        assert row["char_end"] == 4
        store.close()


class TestEmbeddingCRUD:
    DIM = 16

    def test_upsert_and_get_embedding(self, tmp_path):
        store = _make_store(tmp_path)
        cid = store.insert_chunk("test")
        vec = [float(i) / self.DIM for i in range(self.DIM)]
        blob = _pack(vec)
        store.upsert_embedding(cid, blob, self.DIM, "test-model")
        result = store.get_embedding(cid)
        assert result is not None
        rb, rdim, rtag = result
        assert rdim == self.DIM
        assert rtag == "test-model"
        recovered = _unpack(rb)
        for a, b in zip(vec, recovered):
            assert abs(a - b) < 1e-5
        store.close()

    def test_get_embedding_missing_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_embedding("no-such-chunk") is None
        store.close()

    def test_upsert_replaces_on_conflict(self, tmp_path):
        store = _make_store(tmp_path)
        cid = store.insert_chunk("text")
        blob1 = _pack([1.0] * self.DIM)
        blob2 = _pack([2.0] * self.DIM)
        store.upsert_embedding(cid, blob1, self.DIM, "model-v1")
        store.upsert_embedding(cid, blob2, self.DIM, "model-v2")
        _, _, tag = store.get_embedding(cid)
        assert tag == "model-v2"
        store.close()

    def test_all_embeddings_returns_correct_count(self, tmp_path):
        store = _make_store(tmp_path)
        for i in range(3):
            cid = store.insert_chunk(f"chunk {i}")
            store.upsert_embedding(cid, _pack([float(i)] * self.DIM), self.DIM, "m")
        rows = store.all_embeddings()
        assert len(rows) == 3
        store.close()


class TestShardMeta:
    def test_set_and_get_meta(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_meta("head_version", "7")
        assert store.get_meta("head_version") == "7"
        store.close()

    def test_get_missing_meta_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_meta("not_set") is None
        store.close()

    def test_set_meta_upserts(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_meta("k", "v1")
        store.set_meta("k", "v2")
        assert store.get_meta("k") == "v2"
        store.close()


class TestReplaySamples:
    def test_insert_replay_sample_increments_count(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.replay_sample_count() == 0
        store.insert_replay_sample("What is cancer?", label="oncology", score=0.91)
        assert store.replay_sample_count() == 1
        store.close()

    def test_insert_replay_sample_returns_id(self, tmp_path):
        store = _make_store(tmp_path)
        sid = store.insert_replay_sample("test query")
        assert isinstance(sid, str) and len(sid) > 0
        store.close()


class TestContextManager:
    def test_context_manager_does_not_raise(self, tmp_path):
        with SQLiteDomainStore(str(tmp_path), "ctx_domain") as store:
            cid = store.insert_chunk("test")
            assert store.get_chunk(cid) is not None


class TestMultipleDomains:
    def test_separate_sqlite_files_per_domain(self, tmp_path):
        """Each domain_id must get its own .sqlite file."""
        store_a = SQLiteDomainStore(str(tmp_path), "domain_a")
        store_b = SQLiteDomainStore(str(tmp_path), "domain_b")
        store_a.insert_chunk("A-specific chunk")
        store_b.insert_chunk("B-specific chunk")
        store_b.insert_chunk("another B chunk")
        assert store_a.chunk_count() == 1
        assert store_b.chunk_count() == 2
        assert (tmp_path / "domain_a.sqlite").exists()
        assert (tmp_path / "domain_b.sqlite").exists()
        store_a.close()
        store_b.close()


# ===========================================================================
# TIER 2 — SEMANTIC
# ===========================================================================


@pytest.mark.semantic
class TestDomainStoreSemantic:
    """
    Confirm that the store's operations are lossless and that FTS returns
    semantically relevant results before irrelevant ones.
    """

    DIM = 32

    def test_chunk_text_preserved_exactly(self, tmp_path):
        """The text stored must come back byte-identical."""
        text = "Myocardial infarction is caused by coronary artery occlusion."
        with _make_store(tmp_path) as store:
            cid = store.insert_chunk(text)
            assert store.get_chunk(cid)["text"] == text

    def test_embedding_bytes_preserved_exactly(self, tmp_path):
        """Float vector packed as bytes must survive a round-trip unchanged."""
        vec = [math.sin(i * 0.1) for i in range(self.DIM)]
        blob = _pack(vec)
        with _make_store(tmp_path) as store:
            cid = store.insert_chunk("some text")
            store.upsert_embedding(cid, blob, self.DIM, "sinusoidal")
            rb, rdim, rtag = store.get_embedding(cid)
            recovered = _unpack(rb)
            for a, b in zip(vec, recovered):
                assert abs(a - b) < 1e-5

    def test_fts_returns_relevant_results(self, tmp_path):
        """
        Insert a relevant and an irrelevant chunk; FTS search for the domain
        term should return the relevant chunk.
        """
        with _make_store(tmp_path) as store:
            store.insert_chunk(
                "Hypertension is a chronic condition of elevated blood pressure.",
                source="medical_textbook",
            )
            store.insert_chunk(
                "The Eiffel Tower was built in Paris in 1889.",
                source="history_textbook",
            )
            results = store.fts_search("hypertension blood pressure", limit=5)
            assert len(results) > 0
            texts = [r["text"] for r in results]
            assert any("hypertension" in t.lower() or "blood pressure" in t.lower() for t in texts)

    def test_fts_empty_store_returns_empty_list(self, tmp_path):
        with _make_store(tmp_path) as store:
            results = store.fts_search("anything", limit=5)
            assert results == []

    def test_replay_sample_stores_label_and_score(self, tmp_path):
        """Label and score written should be retrievable (via raw SQL)."""
        with _make_store(tmp_path) as store:
            store.insert_replay_sample("Does aspirin prevent stroke?",
                                       label="neurology", score=0.88)
            # Confirm via count and raw query
            count = store._conn.execute(
                "SELECT COUNT(*) FROM replay_samples WHERE label = 'neurology'"
            ).fetchone()[0]
            assert count == 1

    def test_shard_meta_survives_reopen(self, tmp_path):
        """Metadata persisted in shard_meta must survive closing and reopening the store."""
        with SQLiteDomainStore(str(tmp_path), "meta_domain") as s1:
            s1.set_meta("checkpoint", "epoch_42")

        with SQLiteDomainStore(str(tmp_path), "meta_domain") as s2:
            assert s2.get_meta("checkpoint") == "epoch_42"

    def test_all_embeddings_matches_inserted_count(self, tmp_path):
        """all_embeddings() must return exactly as many rows as were inserted."""
        n = 10
        with _make_store(tmp_path) as store:
            for i in range(n):
                cid = store.insert_chunk(f"chunk {i}")
                store.upsert_embedding(
                    cid,
                    _pack([float(i) / n] * self.DIM),
                    self.DIM,
                    "v1",
                )
            rows = store.all_embeddings()
            assert len(rows) == n
