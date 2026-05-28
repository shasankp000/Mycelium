"""
tests/test_evidence_dst_adapter.py
===================================
Unit tests for mycelium.fusion.evidence_dst_adapter.EvidenceDSTAdapter.

Coverage targets
----------------
- fuse() with empty input → vacuous frame, is_genuinely_uncertain=True
- fuse() with single confirmed bundle → m_true dominant
- fuse() with single refuted bundle  → m_false dominant
- MODAL ceiling enforcement          → excess absorbed into m_unknown
- contradiction_trace / stabilization_notes passthrough
- ConflictError fallback path        → conflict_notes populated
- adapt_expert_outputs() idempotency  → existing key never overwritten
- get_evidence_dst_adapter() singleton → same object on repeated calls
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
from types import SimpleNamespace

from mycelium.fusion.evidence_dst_adapter import (
    EvidenceDSTAdapter,
    EvidenceDSTResult,
    adapt_expert_outputs,
    get_evidence_dst_adapter,
)
from mycelium.fusion.dst_fusion import DSTFrame, ConflictError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bundle(
    predicate_id: str = "p1",
    confirmation_score: float = 0.7,
    refutation_score: float = 0.1,
    modal_confidence_ceiling: float | None = None,
    contradiction_trace: list | None = None,
    stabilization_notes: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        predicate_id=predicate_id,
        confirmation_score=confirmation_score,
        refutation_score=refutation_score,
        modal_confidence_ceiling=modal_confidence_ceiling,
        contradiction_trace=contradiction_trace or [],
        stabilization_notes=stabilization_notes or [],
    )


def _make_scored_evidence(bundles: list) -> SimpleNamespace:
    # EvidenceDSTAdapter reads .bundles (not .scored_bundles)
    return SimpleNamespace(bundles=bundles)


# ---------------------------------------------------------------------------
# Tests: fuse() — empty input
# ---------------------------------------------------------------------------

class TestFuseEmpty:
    def test_vacuous_on_empty_bundles(self):
        adapter = EvidenceDSTAdapter()
        result = adapter.fuse(_make_scored_evidence([]))
        assert isinstance(result, EvidenceDSTResult)
        assert result.is_genuinely_uncertain is True
        assert result.net_confidence <= 0.5
        assert result.combined_frame.m_unknown == pytest.approx(1.0)

    def test_vacuous_on_none_bundles(self):
        adapter = EvidenceDSTAdapter()
        result = adapter.fuse(SimpleNamespace(bundles=None))
        assert result.is_genuinely_uncertain is True


# ---------------------------------------------------------------------------
# Tests: fuse() — single bundle
# ---------------------------------------------------------------------------

class TestFuseSingleBundle:
    def test_confirmed_bundle_m_true_dominant(self):
        adapter = EvidenceDSTAdapter()
        bundle = _make_bundle(confirmation_score=0.9, refutation_score=0.05)
        result = adapter.fuse(_make_scored_evidence([bundle]))
        assert result.combined_frame.m_true == pytest.approx(0.9, abs=1e-6)
        assert result.combined_frame.m_false == pytest.approx(0.05, abs=1e-6)
        assert result.net_confidence > 0.7

    def test_refuted_bundle_m_false_dominant(self):
        adapter = EvidenceDSTAdapter()
        bundle = _make_bundle(confirmation_score=0.05, refutation_score=0.9)
        result = adapter.fuse(_make_scored_evidence([bundle]))
        assert result.combined_frame.m_false > result.combined_frame.m_true

    def test_predicate_id_in_per_bundle_frames(self):
        adapter = EvidenceDSTAdapter()
        bundle = _make_bundle(predicate_id="pred_xyz")
        result = adapter.fuse(_make_scored_evidence([bundle]))
        assert "pred_xyz" in result.per_bundle_frames


# ---------------------------------------------------------------------------
# Tests: MODAL ceiling enforcement
# ---------------------------------------------------------------------------

class TestModalCeiling:
    def test_ceiling_caps_m_true_absorbs_into_unknown(self):
        adapter = EvidenceDSTAdapter()
        # confirmation=0.9 but ceiling=0.5 → m_true must be ≤ 0.5
        bundle = _make_bundle(
            confirmation_score=0.9,
            refutation_score=0.0,
            modal_confidence_ceiling=0.5,
        )
        result = adapter.fuse(_make_scored_evidence([bundle]))
        frame = result.combined_frame
        assert frame.m_true <= 0.5 + 1e-6
        assert frame.m_unknown >= 0.35  # excess absorbed into m_unknown, NOT m_false
        assert "p1" in result.modal_ceiling_applied
        assert result.modal_ceiling_applied["p1"] == pytest.approx(0.5)

    def test_no_ceiling_no_entry_in_modal_ceiling_applied(self):
        adapter = EvidenceDSTAdapter()
        bundle = _make_bundle(confirmation_score=0.6)
        result = adapter.fuse(_make_scored_evidence([bundle]))
        assert result.modal_ceiling_applied == {}

    def test_ceiling_not_applied_when_m_true_already_below(self):
        adapter = EvidenceDSTAdapter()
        # ceil=0.5 but confirmation=0.3 → already below, no adjustment
        bundle = _make_bundle(
            confirmation_score=0.3,
            refutation_score=0.0,
            modal_confidence_ceiling=0.5,
        )
        result = adapter.fuse(_make_scored_evidence([bundle]))
        assert result.modal_ceiling_applied == {}


# ---------------------------------------------------------------------------
# Tests: provenance passthrough
# ---------------------------------------------------------------------------

class TestProvenancePassthrough:
    def test_contradiction_trace_aggregated(self):
        adapter = EvidenceDSTAdapter()
        b1 = _make_bundle("p1", contradiction_trace=["[p1] DIRECT sev=0.8 …"])
        b2 = _make_bundle("p2", contradiction_trace=["[p2] PARTIAL sev=0.4 …"])
        result = adapter.fuse(_make_scored_evidence([b1, b2]))
        assert len(result.contradiction_trace) == 2
        assert any("p1" in e for e in result.contradiction_trace)
        assert any("p2" in e for e in result.contradiction_trace)

    def test_stabilization_notes_aggregated(self):
        adapter = EvidenceDSTAdapter()
        b = _make_bundle("p1", stabilization_notes=["treat as upper bound"])
        result = adapter.fuse(_make_scored_evidence([b]))
        assert "treat as upper bound" in result.stabilization_notes


# ---------------------------------------------------------------------------
# Tests: ConflictError fallback
# ---------------------------------------------------------------------------

class TestConflictFallback:
    def test_conflict_error_yields_conflict_note_no_crash(self):
        adapter = EvidenceDSTAdapter()
        b1 = _make_bundle("p1", confirmation_score=0.9, refutation_score=0.0)
        b2 = _make_bundle("p2", confirmation_score=0.0, refutation_score=0.9)

        original_combine = adapter._fusion.combine

        def _raise_conflict(a, b):
            raise ConflictError(k=0.99, frame_a=a, frame_b=b)

        adapter._fusion.combine = _raise_conflict
        result = adapter.fuse(_make_scored_evidence([b1, b2]))
        assert len(result.conflict_notes) >= 1
        assert "ConflictError" in result.conflict_notes[0]
        adapter._fusion.combine = original_combine  # restore


# ---------------------------------------------------------------------------
# Tests: adapt_expert_outputs
# ---------------------------------------------------------------------------

class TestAdaptExpertOutputs:
    def test_injects_evidence_dst_key(self):
        adapter = EvidenceDSTAdapter()
        dst_result = adapter.fuse(_make_scored_evidence([_make_bundle()]))
        outputs = {"existing_key": 42}
        result = adapt_expert_outputs(outputs, dst_result)
        assert "evidence_dst" in result
        assert result["existing_key"] == 42

    def test_idempotent_existing_key_not_overwritten(self):
        adapter = EvidenceDSTAdapter()
        dst_result = adapter.fuse(_make_scored_evidence([_make_bundle()]))
        outputs = {"evidence_dst": "original_value"}
        result = adapt_expert_outputs(outputs, dst_result)
        assert result["evidence_dst"] == "original_value"


# ---------------------------------------------------------------------------
# Tests: singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_same_instance_on_repeated_calls(self):
        a = get_evidence_dst_adapter()
        b = get_evidence_dst_adapter()
        assert a is b


# ---------------------------------------------------------------------------
# Tests: summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_has_required_keys(self):
        adapter = EvidenceDSTAdapter()
        result = adapter.fuse(_make_scored_evidence([_make_bundle()]))
        s = result.summary()
        for key in (
            "net_confidence", "m_true", "m_false", "m_unknown",
            "is_genuinely_uncertain", "predicate_count",
            "modal_ceilings", "conflicts",
        ):
            assert key in s, f"Missing summary key: {key}"
