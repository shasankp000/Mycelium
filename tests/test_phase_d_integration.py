"""
tests/test_phase_d_integration.py
===================================
End-to-end integration test for the full Phase D chain.

Chain
-----
  sentence
    → (stub) PredicatePipeline.run()          – predicate extraction
    → (stub) EvidenceFinder.find()            – retrieve evidence bundles
    → (stub) EvidenceScorer.score()           – score each bundle
    → EvidenceDSTAdapter.fuse()               – DST fusion   [REAL]
    → ContradictionIntegrator.integrate()     – classify pairs [REAL, classify() mocked]

All NLP / embedding calls are replaced with deterministic stubs so the
test runs without model weights.  Only EvidenceDSTAdapter and
ContradictionIntegrator execute real code.

Assertions
----------
1. chain completes without raising
2. dst_result.per_bundle_frames is non-empty
3. dst_result.net_confidence is float in [0, 1]
4. dst_result.modal_ceiling_applied is dict
5. contradiction summary dict has correct keys and types
6. dst_result.contradiction_trace updated when contradictions found
7. predicate_id 'pred_aspirin_01' is in per_bundle_frames
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from mycelium.fusion.evidence_dst_adapter import EvidenceDSTAdapter
from mycelium.pipeline.phase2.contradiction_integrator import ContradictionIntegrator


# ---------------------------------------------------------------------------
# Sentence under test
# ---------------------------------------------------------------------------

SENTENCE = "Aspirin reduces the risk of cardiovascular disease."


# ---------------------------------------------------------------------------
# Deterministic stubs
# ---------------------------------------------------------------------------

def _stub_frame(pred_id: str, negated: bool = False):
    """Build a minimal PredicateFrame-like SimpleNamespace."""
    neg_form = None
    if not negated:
        neg_form = _stub_frame(pred_id, negated=True)
    return SimpleNamespace(
        predicate_id=pred_id,
        surface_form=SENTENCE if not negated else f"NOT: {SENTENCE}",
        negated=negated,
        negated_form=neg_form,
        predicate_type=SimpleNamespace(value="CAUSAL"),
        subject=SimpleNamespace(text="Aspirin"),
        relation=SimpleNamespace(lemma="reduces"),
        object=SimpleNamespace(text="risk of cardiovascular disease"),
        provenance_confidence=0.9,
        modal_certainty=None,
    )


def _stub_evidence_bundle(frame):
    return SimpleNamespace(
        frame=frame,
        source="stub_kb",
        text="Aspirin reduces cardiovascular risk via COX inhibition.",
        relevance=0.82,
    )


def _stub_scored_bundle(eb, pred_id: str):
    """Simulates one ScoredBundle as returned by EvidenceScorer.score()."""
    return SimpleNamespace(
        predicate_id=pred_id,
        bundle=eb,
        confirmation_score=0.78,
        refutation_score=0.08,
        net_confidence=0.74,
        modal_confidence_ceiling=None,
        contradiction_trace=[],
        stabilization_notes=[],
    )


class _StubPredicatePipeline:
    def run(self, sentence: str):
        return SimpleNamespace(
            frames=[_stub_frame("pred_aspirin_01")],
            sentence=sentence,
        )


class _StubEvidenceFinder:
    def find(self, frames, **kwargs):
        return SimpleNamespace(
            bundles=[_stub_evidence_bundle(f) for f in frames],
        )


class _StubEvidenceScorer:
    def score(self, evidence_result, **kwargs):
        scored = [
            _stub_scored_bundle(eb, eb.frame.predicate_id)
            for eb in evidence_result.bundles
        ]
        return SimpleNamespace(
            scored_bundles=scored,
            bundles=scored,          # EvidenceDSTAdapter reads .bundles
            weighted_confidence=0.74,
        )


_CLASSIFY_RESULT = SimpleNamespace(
    contradiction_type="DIRECT_CONTRADICTION",
    severity=0.65,
    confidence=0.88,
    scope="full",
    stage_reached=7,
    explanation="negated predicate semantically inverts original",
)


# ---------------------------------------------------------------------------
# Fixture: run the full chain once, cache results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain_results():
    """Run the full Phase D chain and return (dst_result, summary)."""
    # Stage 1 — predicate extraction
    pipeline    = _StubPredicatePipeline()
    pred_result = pipeline.run(SENTENCE)

    # Stage 2 — evidence finding
    finder    = _StubEvidenceFinder()
    ev_result = finder.find(pred_result.frames)

    # Stage 3 — evidence scoring
    scorer = _StubEvidenceScorer()
    scored = scorer.score(ev_result)

    # Stage 4 — DST fusion (REAL)
    adapter    = EvidenceDSTAdapter()
    dst_result = adapter.fuse(scored)

    # Stage 5 — contradiction integration (REAL, classify mocked)
    integrator = ContradictionIntegrator()
    with patch.object(
        integrator._classifier, "classify", return_value=_CLASSIFY_RESULT
    ):
        summary = integrator.integrate(scored, dst_result=dst_result)

    return dst_result, summary


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPhaseDChain:
    def test_chain_completes_without_error(self, chain_results):
        dst_result, summary = chain_results
        assert dst_result is not None
        assert summary is not None

    def test_per_bundle_frames_non_empty(self, chain_results):
        dst_result, _ = chain_results
        assert len(dst_result.per_bundle_frames) >= 1

    def test_predicate_id_in_per_bundle_frames(self, chain_results):
        dst_result, _ = chain_results
        assert "pred_aspirin_01" in dst_result.per_bundle_frames

    def test_net_confidence_in_unit_interval(self, chain_results):
        dst_result, _ = chain_results
        assert 0.0 <= dst_result.net_confidence <= 1.0

    def test_modal_ceiling_applied_is_dict(self, chain_results):
        dst_result, _ = chain_results
        assert isinstance(dst_result.modal_ceiling_applied, dict)

    def test_summary_has_required_keys(self, chain_results):
        _, summary = chain_results
        required = (
            "classified", "contradictions", "skipped",
            "type_counts", "severity_max", "severity_mean",
        )
        for key in required:
            assert key in summary, f"Missing key in summary: {key}"

    def test_summary_value_types(self, chain_results):
        _, summary = chain_results
        assert isinstance(summary["classified"],     int)
        assert isinstance(summary["contradictions"],  int)
        assert isinstance(summary["type_counts"],    dict)
        assert isinstance(summary["severity_max"],   float)
        assert isinstance(summary["severity_mean"],  float)

    def test_dst_contradiction_trace_updated_when_contradictions_found(
        self, chain_results
    ):
        dst_result, summary = chain_results
        if summary["contradictions"] > 0:
            assert len(dst_result.contradiction_trace) >= 1

    def test_combined_frame_masses_sum_to_one(self, chain_results):
        dst_result, _ = chain_results
        f = dst_result.combined_frame
        total = f.m_true + f.m_false + f.m_unknown
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_per_bundle_frame_masses_sum_to_one(self, chain_results):
        dst_result, _ = chain_results
        for pid, frame in dst_result.per_bundle_frames.items():
            total = frame.m_true + frame.m_false + frame.m_unknown
            assert total == pytest.approx(1.0, abs=1e-5), (
                f"Frame for {pid} masses don't sum to 1: {total}"
            )
