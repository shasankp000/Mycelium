# tests/test_evidence_scorer.py
# Tests for EvidenceScorer (Stage 8).
#
# All tests use hand-built EvidenceResult / EvidenceBundle fixtures.
# No network calls, no LLM calls, no spaCy subprocess.
#
# Run with: pytest tests/test_evidence_scorer.py -v

from __future__ import annotations

import pytest

from mycelium.pipeline.phase2.evidence_scorer import (
    EvidenceScorer,
    ScoreWeights,
    ScoredBundle,
    ScoredEvidenceResult,
    ScoredItem,
    get_evidence_scorer,
)
from mycelium.pipeline.phase2.evidence_types import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceResult,
    RetrievalMethod,
)
from mycelium.pipeline.predicates.predicate_types import (
    EpistemicBurden,
    ModalCertainty,
    PredicateFrame,
    PredicateType,
    QuantifierScope,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_frame(
    predicate_id: str = "p1",
    ptype: PredicateType = PredicateType.FACTIVE,
    falsifiable: bool = True,
    burden: EpistemicBurden = EpistemicBurden.NONE,
    certainty: ModalCertainty | None = None,
) -> PredicateFrame:
    return PredicateFrame(
        predicate_id=predicate_id,
        raw_text="test raw text",
        predicate_type=ptype,
        falsifiable=falsifiable,
        subject="subject",
        predicate_verb="verb",
        object_concept="object",
        negated=False,
        epistemic_burden=burden,
        quantifier_scope=QuantifierScope.UNSCOPED,
        scope_conditions=[],
        modal_certainty=certainty,
        derived_from=[],
        semantic_revision=None,
        domain_hint="general",
        generated_by_phase="test",
    )


def _make_item(
    predicate_id: str = "p1",
    relevance: float = 0.80,
    source_id: str = "doc1",
) -> EvidenceItem:
    return EvidenceItem(
        predicate_id=predicate_id,
        source_id=source_id,
        snippet="some retrieved text",
        relevance_score=relevance,
        retrieval_method=RetrievalMethod.FUSION,
    )


def _make_bundle(
    frame: PredicateFrame,
    items: list[EvidenceItem],
    query: str = "test query",
) -> EvidenceBundle:
    return EvidenceBundle(
        frame=frame,
        items=items,
        query_used=query,
        retrieval_method=RetrievalMethod.FUSION if items else RetrievalMethod.FALLBACK,
        retrieval_ms=1.0,
    )


def _make_result(
    *bundles: EvidenceBundle,
    skipped: list[str] | None = None,
) -> EvidenceResult:
    return EvidenceResult(
        bundles=list(bundles),
        total_ms=10.0,
        store_summary={"total": len(bundles)},
        skipped_ids=skipped or [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoredResultType:
    def test_score_returns_scored_evidence_result(self):
        frame = _make_frame()
        bundle = _make_bundle(frame, [_make_item()])
        result = _make_result(bundle)
        scored = EvidenceScorer().score(result)
        assert isinstance(scored, ScoredEvidenceResult)

    def test_scored_bundles_count_matches_input_bundles(self):
        b1 = _make_bundle(_make_frame("p1"), [_make_item("p1")])
        b2 = _make_bundle(_make_frame("p2"), [_make_item("p2")])
        scored = EvidenceScorer().score(_make_result(b1, b2))
        assert len(scored.scored_bundles) == 2


class TestCompositeScoreFormula:
    def test_factive_none_burden_composite_equals_raw(self):
        """FACTIVE + NONE burden: composite = raw * 1.0 * 1.0"""
        frame = _make_frame(burden=EpistemicBurden.NONE)
        item = _make_item(relevance=0.80)
        bundle = _make_bundle(frame, [item])
        scored = EvidenceScorer().score(_make_result(bundle))
        si = scored.scored_bundles[0].scored_items[0]
        assert abs(si.composite_score - 0.80) < 1e-9

    def test_scope_bounded_burden_applies_0_85_multiplier(self):
        frame = _make_frame(burden=EpistemicBurden.SCOPE_BOUNDED)
        item = _make_item(relevance=1.0)
        bundle = _make_bundle(frame, [item])
        scored = EvidenceScorer().score(_make_result(bundle))
        si = scored.scored_bundles[0].scored_items[0]
        assert abs(si.composite_score - 0.85) < 1e-9

    def test_rebuttal_burden_applies_1_10_multiplier_clamped(self):
        """REBUTTAL * 1.1 can exceed 1.0 -> clamped to 1.0"""
        frame = _make_frame(burden=EpistemicBurden.REBUTTAL)
        item = _make_item(relevance=1.0)
        bundle = _make_bundle(frame, [item])
        scored = EvidenceScorer().score(_make_result(bundle))
        si = scored.scored_bundles[0].scored_items[0]
        assert si.composite_score == 1.0  # clamped

    def test_rebuttal_burden_boosts_score_below_cap(self):
        frame = _make_frame(burden=EpistemicBurden.REBUTTAL)
        item = _make_item(relevance=0.80)
        bundle = _make_bundle(frame, [item])
        scored = EvidenceScorer().score(_make_result(bundle))
        si = scored.scored_bundles[0].scored_items[0]
        assert abs(si.composite_score - 0.88) < 1e-6

    def test_exhaustive_burden_applies_0_90_multiplier(self):
        frame = _make_frame(burden=EpistemicBurden.EXHAUSTIVE)
        item = _make_item(relevance=1.0)
        bundle = _make_bundle(frame, [item])
        scored = EvidenceScorer().score(_make_result(bundle))
        si = scored.scored_bundles[0].scored_items[0]
        assert abs(si.composite_score - 0.90) < 1e-9

    def test_modal_burden_applies_0_70_multiplier(self):
        frame = _make_frame(
            ptype=PredicateType.MODAL,
            burden=EpistemicBurden.MODAL,
            certainty=ModalCertainty.CERTAIN,
        )
        item = _make_item(relevance=1.0)
        bundle = _make_bundle(frame, [item])
        scored = EvidenceScorer().score(_make_result(bundle))
        si = scored.scored_bundles[0].scored_items[0]
        # modal burden 0.70 * certainty_certain 1.0
        assert abs(si.composite_score - 0.70) < 1e-9


class TestModalCertaintyMultiplier:
    def _score_modal(self, certainty: ModalCertainty, relevance: float = 1.0) -> float:
        frame = _make_frame(
            ptype=PredicateType.MODAL,
            burden=EpistemicBurden.NONE,  # isolate certainty only
            certainty=certainty,
        )
        item = _make_item(relevance=relevance)
        bundle = _make_bundle(frame, [item])
        scored = EvidenceScorer().score(_make_result(bundle))
        return scored.scored_bundles[0].scored_items[0].composite_score

    def test_certain_multiplier_is_1_0(self):
        assert abs(self._score_modal(ModalCertainty.CERTAIN) - 1.0) < 1e-9

    def test_probable_multiplier_is_0_9(self):
        assert abs(self._score_modal(ModalCertainty.PROBABLE) - 0.9) < 1e-9

    def test_possible_multiplier_is_0_75(self):
        assert abs(self._score_modal(ModalCertainty.POSSIBLE) - 0.75) < 1e-9

    def test_speculative_multiplier_is_0_6(self):
        assert abs(self._score_modal(ModalCertainty.SPECULATIVE) - 0.6) < 1e-9


class TestNormativeExcluded:
    def test_normative_bundle_produces_zero_scored_bundles(self):
        """EvidenceFinder skips NORMATIVE; if passed directly, scorer handles gracefully."""
        norm_frame = _make_frame(
            "p_norm", ptype=PredicateType.NORMATIVE, falsifiable=False
        )
        bundle = _make_bundle(norm_frame, [])
        result = _make_result(bundle, skipped=["p_norm"])
        scored = EvidenceScorer().score(result)
        # Bundle is carried through (empty items -> confidence 0)
        assert scored.scored_bundles[0].bundle_confidence == 0.0
        assert scored.scored_bundles[0].scored_items == []


class TestBundleConfidence:
    def test_bundle_confidence_is_mean_of_positive_composites(self):
        frame = _make_frame(burden=EpistemicBurden.NONE)
        items = [
            _make_item(relevance=0.80, source_id="a"),
            _make_item(relevance=0.60, source_id="b"),
        ]
        bundle = _make_bundle(frame, items)
        scored = EvidenceScorer().score(_make_result(bundle))
        sb = scored.scored_bundles[0]
        expected = (0.80 + 0.60) / 2
        assert abs(sb.bundle_confidence - expected) < 1e-9

    def test_empty_bundle_confidence_is_zero(self):
        frame = _make_frame()
        bundle = _make_bundle(frame, [])
        scored = EvidenceScorer().score(_make_result(bundle))
        assert scored.scored_bundles[0].bundle_confidence == 0.0


class TestWeightedConfidence:
    def test_weighted_confidence_is_item_count_weighted_mean(self):
        frame1 = _make_frame("p1", burden=EpistemicBurden.NONE)
        frame2 = _make_frame("p2", burden=EpistemicBurden.NONE)
        # bundle1: 1 item at 0.9, bundle2: 2 items at 0.5 each
        b1 = _make_bundle(frame1, [_make_item("p1", 0.9)])
        b2 = _make_bundle(frame2, [
            _make_item("p2", 0.5, "a"),
            _make_item("p2", 0.5, "b"),
        ])
        scored = EvidenceScorer().score(_make_result(b1, b2))
        # bundle1 confidence=0.9 weight=1, bundle2 confidence=0.5 weight=2
        expected = (0.9 * 1 + 0.5 * 2) / 3
        assert abs(scored.weighted_confidence - expected) < 1e-9

    def test_all_empty_bundles_weighted_confidence_is_zero(self):
        frame = _make_frame()
        scored = EvidenceScorer().score(_make_result(_make_bundle(frame, [])))
        assert scored.weighted_confidence == 0.0


class TestScoredItemOrdering:
    def test_scored_items_sorted_descending_by_composite(self):
        frame = _make_frame(burden=EpistemicBurden.NONE)
        items = [
            _make_item(relevance=0.3, source_id="low"),
            _make_item(relevance=0.9, source_id="high"),
            _make_item(relevance=0.6, source_id="mid"),
        ]
        bundle = _make_bundle(frame, items)
        scored = EvidenceScorer().score(_make_result(bundle))
        scores = [s.composite_score for s in scored.scored_bundles[0].scored_items]
        assert scores == sorted(scores, reverse=True)


class TestScoreBreakdown:
    def test_breakdown_contains_required_keys(self):
        frame = _make_frame()
        bundle = _make_bundle(frame, [_make_item()])
        scored = EvidenceScorer().score(_make_result(bundle))
        bd = scored.scored_bundles[0].scored_items[0].score_breakdown
        for key in ("raw_score", "burden_multiplier", "certainty_multiplier",
                    "composite_score", "burden", "certainty"):
            assert key in bd

    def test_breakdown_composite_matches_scored_item_composite(self):
        frame = _make_frame(burden=EpistemicBurden.SCOPE_BOUNDED)
        bundle = _make_bundle(frame, [_make_item(relevance=0.80)])
        scored = EvidenceScorer().score(_make_result(bundle))
        si = scored.scored_bundles[0].scored_items[0]
        assert si.score_breakdown["composite_score"] == si.composite_score

    def test_custom_weights_reflected_in_breakdown(self):
        weights = ScoreWeights(burden_scope_bounded=0.50)
        scorer = EvidenceScorer(weights=weights)
        frame = _make_frame(burden=EpistemicBurden.SCOPE_BOUNDED)
        bundle = _make_bundle(frame, [_make_item(relevance=1.0)])
        scored = scorer.score(_make_result(bundle))
        si = scored.scored_bundles[0].scored_items[0]
        assert abs(si.composite_score - 0.50) < 1e-9


class TestSingleton:
    def test_get_evidence_scorer_returns_singleton(self):
        a = get_evidence_scorer()
        b = get_evidence_scorer()
        assert a is b

    def test_singleton_is_evidence_scorer_instance(self):
        assert isinstance(get_evidence_scorer(), EvidenceScorer)
