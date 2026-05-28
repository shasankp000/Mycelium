"""
tests/test_contradiction_integrator.py
=======================================
Unit tests for mycelium.pipeline.phase2.contradiction_integrator.

Coverage targets
----------------
- integrate() with no negated_form    → all skipped, classified=0
- integrate() with negated_form       → classified incremented
- Significant contradiction type      → bundle.contradiction_trace populated
- Non-significant (NON_CONTRADICTORY_DIVERGENCE) → trace NOT populated
- dst_result propagation              → contradiction_trace on DST also updated
- ScoredBundle missing trace fields   → fields patched in (forward compat)
- classify() exception handling       → graceful skip, no crash
- type_counts aggregated across bundles
- Singleton accessor                  → same object on repeated calls
- PredicateFrameNode label / canonical_form / embedding_signature
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
from types import SimpleNamespace

from mycelium.pipeline.phase2.contradiction_integrator import (
    ContradictionIntegrator,
    get_contradiction_integrator,
    PredicateFrameNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(
    predicate_id: str = "p1",
    surface_form: str = "A causes B",
    negated: bool = False,
    negated_form=None,
    predicate_type_value: str = "CAUSAL",
    provenance_confidence: float = 0.85,
):
    subject  = SimpleNamespace(text="A")
    relation = SimpleNamespace(lemma="causes")
    obj      = SimpleNamespace(text="B")
    pt       = SimpleNamespace(value=predicate_type_value)
    return SimpleNamespace(
        predicate_id=predicate_id,
        surface_form=surface_form,
        negated=negated,
        negated_form=negated_form,
        predicate_type=pt,
        subject=subject,
        relation=relation,
        object=obj,
        provenance_confidence=provenance_confidence,
    )


def _make_bundle(frame=None, negated_frame=None):
    if frame is None:
        frame = _make_frame(negated_form=negated_frame)
    inner = SimpleNamespace(frame=frame)
    b = SimpleNamespace(bundle=inner)
    b.contradiction_trace = []
    b.stabilization_notes = []
    return b


def _make_scored_evidence(bundles: list):
    return SimpleNamespace(scored_bundles=bundles)


def _mock_result(
    contradiction_type: str = "DIRECT_CONTRADICTION",
    severity: float = 0.75,
    confidence: float = 0.9,
    scope: str = "full",
    stage_reached: int = 7,
    explanation: str = "test contradiction",
):
    return SimpleNamespace(
        contradiction_type=contradiction_type,
        severity=severity,
        confidence=confidence,
        scope=scope,
        stage_reached=stage_reached,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Tests: no negated_form
# ---------------------------------------------------------------------------

class TestNoNegatedForm:
    def test_all_skipped_when_no_negated_form(self):
        integrator = ContradictionIntegrator()
        bundle = _make_bundle(_make_frame(negated_form=None))
        summary = integrator.integrate(_make_scored_evidence([bundle]))
        assert summary["skipped"] == 1
        assert summary["classified"] == 0
        assert summary["contradictions"] == 0

    def test_empty_scored_bundles(self):
        integrator = ContradictionIntegrator()
        summary = integrator.integrate(_make_scored_evidence([]))
        assert summary["classified"] == 0
        assert summary["skipped"] == 0


# ---------------------------------------------------------------------------
# Tests: with negated_form
# ---------------------------------------------------------------------------

class TestWithNegatedForm:
    def test_classified_incremented(self):
        integrator = ContradictionIntegrator()
        neg    = _make_frame("p1", negated=True)
        frame  = _make_frame("p1", negated_form=neg)
        bundle = _make_bundle(frame)
        with patch.object(integrator._classifier, "classify",
                          return_value=_mock_result("DIRECT_CONTRADICTION", severity=0.8)):
            summary = integrator.integrate(_make_scored_evidence([bundle]))
        assert summary["classified"] == 1
        assert summary["contradictions"] == 1
        assert summary["severity_max"] == pytest.approx(0.8)

    def test_contradiction_trace_populated(self):
        integrator = ContradictionIntegrator()
        neg    = _make_frame("p1", negated=True)
        frame  = _make_frame("p1", negated_form=neg)
        bundle = _make_bundle(frame)
        with patch.object(integrator._classifier, "classify",
                          return_value=_mock_result("DIRECT_CONTRADICTION", severity=0.8)):
            integrator.integrate(_make_scored_evidence([bundle]))
        assert len(bundle.contradiction_trace) == 1
        assert "DIRECT_CONTRADICTION" in bundle.contradiction_trace[0]
        assert len(bundle.stabilization_notes) == 1

    def test_non_significant_type_not_in_trace(self):
        integrator = ContradictionIntegrator()
        neg    = _make_frame("p1", negated=True)
        frame  = _make_frame("p1", negated_form=neg)
        bundle = _make_bundle(frame)
        with patch.object(integrator._classifier, "classify",
                          return_value=_mock_result("NON_CONTRADICTORY_DIVERGENCE", severity=0.1)):
            summary = integrator.integrate(_make_scored_evidence([bundle]))
        assert len(bundle.contradiction_trace) == 0
        assert summary["contradictions"] == 0
        assert summary["classified"] == 1  # classified but not significant


# ---------------------------------------------------------------------------
# Tests: forward-compat field patching
# ---------------------------------------------------------------------------

class TestForwardCompatFieldPatching:
    def test_missing_trace_fields_patched_in(self):
        integrator = ContradictionIntegrator()
        inner = SimpleNamespace(frame=_make_frame(negated_form=None))
        bundle = SimpleNamespace(bundle=inner)  # deliberately no trace fields
        assert not hasattr(bundle, "contradiction_trace")
        integrator.integrate(_make_scored_evidence([bundle]))
        assert hasattr(bundle, "contradiction_trace")
        assert hasattr(bundle, "stabilization_notes")


# ---------------------------------------------------------------------------
# Tests: dst_result propagation
# ---------------------------------------------------------------------------

class TestDSTResultPropagation:
    def test_dst_result_trace_updated(self):
        integrator = ContradictionIntegrator()
        neg    = _make_frame("p1", negated=True)
        frame  = _make_frame("p1", negated_form=neg)
        bundle = _make_bundle(frame)
        dst_result = SimpleNamespace(contradiction_trace=[], stabilization_notes=[])
        with patch.object(integrator._classifier, "classify",
                          return_value=_mock_result("DIRECT_CONTRADICTION")):
            integrator.integrate(_make_scored_evidence([bundle]), dst_result=dst_result)
        assert len(dst_result.contradiction_trace) == 1
        assert len(dst_result.stabilization_notes) == 1


# ---------------------------------------------------------------------------
# Tests: classify() exception handling
# ---------------------------------------------------------------------------

class TestClassifyExceptionHandling:
    def test_classify_exception_gracefully_skipped(self):
        integrator = ContradictionIntegrator()
        neg    = _make_frame("p1", negated=True)
        frame  = _make_frame("p1", negated_form=neg)
        bundle = _make_bundle(frame)
        with patch.object(integrator._classifier, "classify",
                          side_effect=RuntimeError("boom")):
            summary = integrator.integrate(_make_scored_evidence([bundle]))
        assert summary["skipped"] == 1
        assert summary["classified"] == 0


# ---------------------------------------------------------------------------
# Tests: type_counts
# ---------------------------------------------------------------------------

class TestTypeCounts:
    def test_type_counts_aggregated(self):
        integrator = ContradictionIntegrator()
        bundles = []
        for i in range(3):
            neg  = _make_frame(f"p{i}", negated=True)
            frm  = _make_frame(f"p{i}", negated_form=neg)
            bundles.append(_make_bundle(frm))
        with patch.object(integrator._classifier, "classify",
                          return_value=_mock_result("PARTIAL_CONTRADICTION", severity=0.5)):
            summary = integrator.integrate(_make_scored_evidence(bundles))
        assert summary["type_counts"].get("PARTIAL_CONTRADICTION") == 3
        assert summary["contradictions"] == 3

    def test_severity_mean_correct(self):
        integrator = ContradictionIntegrator()
        call_count = 0
        severities = [0.4, 0.6]

        def _side_effect(a, b):
            nonlocal call_count
            sev = severities[call_count % len(severities)]
            call_count += 1
            return _mock_result("DIRECT_CONTRADICTION", severity=sev)

        bundles = []
        for i in range(2):
            neg  = _make_frame(f"p{i}", negated=True)
            frm  = _make_frame(f"p{i}", negated_form=neg)
            bundles.append(_make_bundle(frm))
        with patch.object(integrator._classifier, "classify", side_effect=_side_effect):
            summary = integrator.integrate(_make_scored_evidence(bundles))
        assert summary["severity_mean"] == pytest.approx(0.5, abs=0.001)


# ---------------------------------------------------------------------------
# Tests: singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_same_instance_on_repeated_calls(self):
        a = get_contradiction_integrator()
        b = get_contradiction_integrator()
        assert a is b


# ---------------------------------------------------------------------------
# Tests: PredicateFrameNode
# ---------------------------------------------------------------------------

class TestPredicateFrameNode:
    def test_label_from_surface_form(self):
        frame = _make_frame(surface_form="X implies Y")
        node  = PredicateFrameNode(frame)
        assert node.label == "X implies Y"

    def test_canonical_form_contains_family_and_parts(self):
        frame = _make_frame()
        node  = PredicateFrameNode(frame)
        cf = node.semantic_signature.canonical_form
        assert "::" in cf
        assert "CAUSAL" in cf
        assert "A" in cf
        assert "B" in cf

    def test_embedding_signature_is_empty_list(self):
        frame = _make_frame()
        node  = PredicateFrameNode(frame)
        assert node.semantic_signature.embedding_signature == []

    def test_temporal_unknown_disables_temporal_gate(self):
        frame = _make_frame()
        node  = PredicateFrameNode(frame)
        assert node.temporal_state.type == "UNKNOWN"

    def test_negated_frame_has_historical_validity_true(self):
        frame = _make_frame(negated=True)
        node  = PredicateFrameNode(frame)
        assert node.temporal_state.historical_validity is True
