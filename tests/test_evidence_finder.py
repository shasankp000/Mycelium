# tests/test_evidence_finder.py
# Tests for EvidenceFinder (Stage 7).
#
# All tests use injected stub retrievers.
# No network calls, no LLM calls, no spaCy subprocess.
#
# Run with: pytest tests/test_evidence_finder.py -v

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from mycelium.pipeline.phase2.evidence_finder import (
    EvidenceFinder,
    _build_query,
    get_evidence_finder,
)
from mycelium.pipeline.phase2.evidence_types import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceResult,
    RetrievalMethod,
)
from mycelium.pipeline.predicates.predicate_store import PredicateStore
from mycelium.pipeline.predicates.predicate_types import (
    EpistemicBurden,
    PredicateFrame,
    PredicateType,
    QuantifierScope,
)


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

class _StubRetriever:
    """Returns a fixed list of result dicts. Tracks call count."""

    def __init__(self, results: List[Dict[str, Any]]) -> None:
        self._results = results
        self.call_count = 0
        self.last_query: str = ""

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self.call_count += 1
        self.last_query = query_text
        return self._results[:top_k]


class _RaisingRetriever:
    """Always raises RuntimeError to test graceful fallback."""

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        raise RuntimeError("Simulated retriever failure")


def _make_frame(
    predicate_id: str,
    ptype: PredicateType = PredicateType.FACTIVE,
    falsifiable: bool = True,
    raw_text: str = "Paris is the capital of France",
    subject: str = "Paris",
    predicate_verb: str = "is",
    object_concept: str = "capital of France",
) -> PredicateFrame:
    return PredicateFrame(
        predicate_id=predicate_id,
        raw_text=raw_text,
        predicate_type=ptype,
        falsifiable=falsifiable,
        subject=subject,
        predicate_verb=predicate_verb,
        object_concept=object_concept,
        negated=False,
        epistemic_burden=EpistemicBurden.NONE,
        quantifier_scope=QuantifierScope.UNSCOPED,
        scope_conditions=[],
        modal_certainty=None,
        derived_from=[],
        semantic_revision=None,
        domain_hint="general",
        generated_by_phase="test",
    )


def _make_store(*frames: PredicateFrame) -> PredicateStore:
    store = PredicateStore()
    for f in frames:
        store.add(f)
    return store


_SAMPLE_RESULTS = [
    {"id": "doc1", "snippet": "Paris has been the French capital since 987.",
     "relevance": 0.92, "url": "https://example.com/1", "title": "Paris"},
    {"id": "doc2", "snippet": "France is a unitary republic.",
     "relevance": 0.75, "url": "https://example.com/2", "title": "France"},
    {"id": "doc3", "snippet": "The French government is located in Paris.",
     "relevance": 0.61, "url": "https://example.com/3", "title": "Government"},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvidenceResultType:
    def test_find_returns_evidence_result(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        assert isinstance(result, EvidenceResult)

    def test_result_has_store_summary(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        assert isinstance(result.store_summary, dict)
        assert "total" in result.store_summary


class TestBundlePerFalsifiableFrame:
    def test_one_bundle_per_falsifiable_frame(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(
            _make_frame("p1"),
            _make_frame("p2"),
        )
        result = finder.find(store)
        assert len(result.bundles) == 2

    def test_bundle_contains_evidence_items(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        assert len(result.bundles[0].items) == len(_SAMPLE_RESULTS)

    def test_bundle_items_are_evidence_item_instances(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        for item in result.bundles[0].items:
            assert isinstance(item, EvidenceItem)


class TestNormativeFramesSkipped:
    def test_normative_frame_produces_no_bundle(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        normative = _make_frame(
            "p_norm", ptype=PredicateType.NORMATIVE, falsifiable=False
        )
        store = _make_store(normative)
        result = finder.find(store)
        assert len(result.bundles) == 0

    def test_normative_frame_id_in_skipped(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        normative = _make_frame(
            "p_norm", ptype=PredicateType.NORMATIVE, falsifiable=False
        )
        store = _make_store(normative)
        result = finder.find(store)
        assert "p_norm" in result.skipped_ids

    def test_mixed_store_skips_only_normative(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        factive = _make_frame("p_fact")
        normative = _make_frame(
            "p_norm", ptype=PredicateType.NORMATIVE, falsifiable=False
        )
        store = _make_store(factive, normative)
        result = finder.find(store)
        assert len(result.bundles) == 1
        assert result.bundles[0].frame.predicate_id == "p_fact"


class TestQueryConstruction:
    def test_query_used_is_non_empty(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        assert result.bundles[0].query_used != ""

    def test_query_uses_subject_and_predicate_verb(self):
        frame = _make_frame(
            "p1", subject="nuclear power",
            predicate_verb="causes", object_concept="deaths"
        )
        query = _build_query(frame)
        assert "nuclear power" in query
        assert "causes" in query

    def test_query_fallback_uses_raw_text_when_fields_empty(self):
        frame = _make_frame(
            "p1",
            raw_text="vaccines reduce mortality rates in children",
            subject="", predicate_verb="", object_concept="",
        )
        query = _build_query(frame)
        assert "vaccines" in query


class TestRelevanceOrdering:
    def test_items_sorted_descending_by_relevance(self):
        unordered = [
            {"id": "a", "snippet": "text a", "relevance": 0.4},
            {"id": "b", "snippet": "text b", "relevance": 0.9},
            {"id": "c", "snippet": "text c", "relevance": 0.6},
        ]
        stub = _StubRetriever(unordered)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        scores = [i.relevance_score for i in result.bundles[0].items]
        assert scores == sorted(scores, reverse=True)

    def test_best_returns_highest_relevance_item(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        bundle = result.bundles[0]
        assert bundle.best().relevance_score == max(
            i.relevance_score for i in bundle.items
        )


class TestFallbackOnRetrieverError:
    def test_raising_primary_produces_empty_bundle_not_exception(self):
        finder = EvidenceFinder(primary_retriever=_RaisingRetriever())
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)  # must not raise
        assert result.bundles[0].items == []

    def test_raising_primary_sets_fallback_method(self):
        finder = EvidenceFinder(
            primary_retriever=_RaisingRetriever(),
            fallback_retriever=None,
        )
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        assert result.bundles[0].retrieval_method == RetrievalMethod.FALLBACK

    def test_primary_error_falls_through_to_fallback_retriever(self):
        fallback = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(
            primary_retriever=_RaisingRetriever(),
            fallback_retriever=fallback,
        )
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        assert len(result.bundles[0].items) == len(_SAMPLE_RESULTS)


class TestTiming:
    def test_bundle_retrieval_ms_is_positive(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        assert result.bundles[0].retrieval_ms > 0.0

    def test_result_total_ms_is_positive(self):
        stub = _StubRetriever(_SAMPLE_RESULTS)
        finder = EvidenceFinder(primary_retriever=stub)
        store = _make_store(_make_frame("p1"))
        result = finder.find(store)
        assert result.total_ms > 0.0


class TestSingleton:
    def test_get_evidence_finder_returns_singleton(self):
        a = get_evidence_finder()
        b = get_evidence_finder()
        assert a is b

    def test_singleton_is_evidence_finder_instance(self):
        assert isinstance(get_evidence_finder(), EvidenceFinder)
