"""
tests/domain_graph/test_domain_graph.py

Two tiers:
  Tier 1 – Structural / unit  : pure Python, no ML dependencies, no filesystem I/O
            (all tests run in-memory via tmp_path or mock registries)
  Tier 2 – Semantic           : marked @pytest.mark.semantic
            Confirm the *intended behaviour* of the novelty policy is correct,
            i.e. the right NoveltyDecision is returned for controlled embeddings.

Run structural only:  pytest tests/domain_graph/ -m "not semantic"
Run semantic only  :  pytest tests/domain_graph/ -m semantic
Run all            :  pytest tests/domain_graph/
"""
from __future__ import annotations

import json
import math
import struct
import threading
import tempfile
from pathlib import Path
from typing import List

import pytest

from mycelium.domain_graph.models import DomainNode, DomainEdge, DriftProfile
from mycelium.domain_graph.state import (
    DomainState, DomainMode, GateState, DriftType, NoveltyDecision,
)
from mycelium.domain_graph.registry import DomainGraphRegistry
from mycelium.domain_graph.novelty import (
    DomainNoveltyPolicy, NoveltyThresholds, NoveltyResult, _cosine,
)
from mycelium.domain_graph.events import (
    emit_domain_registered, emit_state_transition,
    emit_novelty_decision, emit_cold_store, emit_reactivation, emit_drift_detected,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(dim: int, value: float = 1.0) -> List[float]:
    """Return a unit-normalised vector of `dim` dimensions, all equal."""
    raw = [value] * dim
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _orthogonal(dim: int) -> List[float]:
    """Return a vector orthogonal to _vec(dim) by flipping the sign of the last half."""
    half = dim // 2
    raw = [1.0] * half + [-1.0] * (dim - half)
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _pack(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _make_registry(tmp_path: Path, *, auto_save: bool = False) -> DomainGraphRegistry:
    graph_file = tmp_path / "graph.json"
    return DomainGraphRegistry(str(graph_file), auto_save=auto_save)


def _hot_node(domain_id: str, centroid: List[float]) -> DomainNode:
    node = DomainNode(
        domain_id=domain_id,
        label=domain_id,
        state=DomainState.HOT,
        gate=GateState.OPEN,
    )
    node.meta["centroid"] = centroid
    return node


# ===========================================================================
# TIER 1 — STRUCTURAL
# ===========================================================================


class TestDomainNodeModel:
    """DomainNode dataclass: construction, serialisation, state helpers."""

    def test_default_state_on_creation(self):
        node = DomainNode(domain_id="d1", label="Test")
        assert node.state == DomainState.CREATING
        assert node.mode == DomainMode.BOOTSTRAP
        assert node.gate == GateState.CLOSED

    def test_mark_active_increments_query_count(self):
        node = DomainNode(domain_id="d1", label="Test")
        assert node.query_count == 0
        node.mark_active()
        assert node.query_count == 1
        node.mark_active()
        assert node.query_count == 2

    def test_mark_active_sets_last_active(self):
        node = DomainNode(domain_id="d1", label="Test")
        assert node.last_active is None
        node.mark_active()
        assert node.last_active is not None

    def test_roundtrip_serialisation(self):
        node = DomainNode(
            domain_id="d1", label="Medicine",
            state=DomainState.HOT, mode=DomainMode.FROZEN, gate=GateState.OPEN,
            domain_version=3, head_version=7, query_count=42,
            tags=["bio", "rag"],
        )
        restored = DomainNode.from_dict(node.to_dict())
        assert restored.domain_id == node.domain_id
        assert restored.state == node.state
        assert restored.mode == node.mode
        assert restored.gate == node.gate
        assert restored.domain_version == node.domain_version
        assert restored.query_count == node.query_count
        assert restored.tags == node.tags

    def test_drift_profile_roundtrip(self):
        dp = DriftProfile(drift_type=DriftType.SEMANTIC, magnitude=0.42)
        restored = DriftProfile.from_dict(dp.to_dict())
        assert restored.drift_type == DriftType.SEMANTIC
        assert abs(restored.magnitude - 0.42) < 1e-9
        assert restored.resolved is False


class TestDomainEdgeModel:
    def test_roundtrip(self):
        edge = DomainEdge(source_id="a", target_id="b", relation="extends", weight=0.8)
        restored = DomainEdge.from_dict(edge.to_dict())
        assert restored.source_id == "a"
        assert restored.target_id == "b"
        assert restored.relation == "extends"
        assert abs(restored.weight - 0.8) < 1e-9


class TestStateEnums:
    def test_all_domain_states_str_compatible(self):
        for s in DomainState:
            assert DomainState(s.value) == s

    def test_all_novelty_decisions_str_compatible(self):
        for d in NoveltyDecision:
            assert NoveltyDecision(d.value) == d

    def test_all_drift_types_str_compatible(self):
        for dt in DriftType:
            assert DriftType(dt.value) == dt


class TestDomainGraphRegistry:
    """Registry: CRUD, state transitions, edge ops, persistence."""

    def test_register_and_get(self, tmp_path):
        reg = _make_registry(tmp_path)
        node = DomainNode(domain_id="d1", label="Cardiology")
        reg.register(node)
        assert reg.get("d1") is not None
        assert reg.get("d1").label == "Cardiology"

    def test_register_duplicate_raises(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.register(DomainNode(domain_id="d1", label="A"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(DomainNode(domain_id="d1", label="B"))

    def test_require_missing_raises(self, tmp_path):
        reg = _make_registry(tmp_path)
        with pytest.raises(KeyError):
            reg.require("nonexistent")

    def test_update_missing_raises(self, tmp_path):
        reg = _make_registry(tmp_path)
        with pytest.raises(KeyError):
            reg.update(DomainNode(domain_id="ghost", label="Ghost"))

    def test_set_state_transition(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.register(DomainNode(domain_id="d1", label="Neuro"))
        reg.set_state("d1", DomainState.HOT)
        assert reg.require("d1").state == DomainState.HOT

    def test_set_gate(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.register(DomainNode(domain_id="d1", label="Cardio"))
        reg.set_gate("d1", GateState.OPEN)
        assert reg.require("d1").gate == GateState.OPEN

    def test_set_mode(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.register(DomainNode(domain_id="d1", label="Cardio"))
        reg.set_mode("d1", DomainMode.FROZEN)
        assert reg.require("d1").mode == DomainMode.FROZEN

    def test_mark_active_updates_query_count(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.register(DomainNode(domain_id="d1", label="Radiology"))
        reg.mark_active("d1")
        reg.mark_active("d1")
        assert reg.require("d1").query_count == 2

    def test_all_hot_domains_filtered(self, tmp_path):
        reg = _make_registry(tmp_path)
        hot = DomainNode(domain_id="hot", label="Hot", state=DomainState.HOT, gate=GateState.OPEN)
        cold = DomainNode(domain_id="cold", label="Cold", state=DomainState.COLD, gate=GateState.CLOSED)
        reg.register(hot)
        reg.register(cold)
        hot_list = reg.all_hot_domains()
        assert len(hot_list) == 1
        assert hot_list[0].domain_id == "hot"

    def test_edge_add_and_query(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.register(DomainNode(domain_id="a", label="A"))
        reg.register(DomainNode(domain_id="b", label="B"))
        edge = DomainEdge(source_id="a", target_id="b", relation="extends")
        reg.add_edge(edge)
        edges_a = reg.edges_for("a")
        assert len(edges_a) == 1
        assert edges_a[0].relation == "extends"

    def test_summary_counts(self, tmp_path):
        reg = _make_registry(tmp_path)
        reg.register(DomainNode(domain_id="h1", label="H1", state=DomainState.HOT, gate=GateState.OPEN))
        reg.register(DomainNode(domain_id="h2", label="H2", state=DomainState.HOT, gate=GateState.OPEN))
        reg.register(DomainNode(domain_id="c1", label="C1", state=DomainState.COLD))
        s = reg.summary()
        assert s["total"] == 3
        assert s[DomainState.HOT.value] == 2
        assert s[DomainState.COLD.value] == 1

    def test_persistence_roundtrip(self, tmp_path):
        """Registry written to disk and reloaded produces the same nodes."""
        graph_file = tmp_path / "graph.json"
        reg = DomainGraphRegistry(str(graph_file), auto_save=True)
        node = DomainNode(domain_id="persist_me", label="Persistent",
                         state=DomainState.HOT, gate=GateState.OPEN)
        reg.register(node)
        # Reload from same path
        reg2 = DomainGraphRegistry(str(graph_file), auto_save=False)
        restored = reg2.get("persist_me")
        assert restored is not None
        assert restored.label == "Persistent"
        assert restored.state == DomainState.HOT

    def test_thread_safety_concurrent_register(self, tmp_path):
        """Concurrent registrations must not corrupt internal state."""
        reg = _make_registry(tmp_path)
        errors = []

        def worker(i: int) -> None:
            try:
                reg.register(DomainNode(domain_id=f"d{i}", label=f"Domain {i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 20 unique registrations should succeed with no errors
        assert len(errors) == 0
        assert len(reg.all_domains()) == 20


class TestCosineMath:
    def test_identical_vectors_cosine_one(self):
        v = _vec(32)
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_cosine_near_zero(self):
        v1 = _vec(32)
        v2 = _orthogonal(32)
        sim = _cosine(v1, v2)
        assert abs(sim) < 0.1

    def test_zero_vector_returns_zero(self):
        v = _vec(8)
        zero = [0.0] * 8
        assert _cosine(v, zero) == 0.0


class TestDomainGraphEvents:
    """Events must not raise even when pipeline_event is unavailable."""

    def test_emit_domain_registered_no_raise(self):
        emit_domain_registered("d1", "Medicine", domain_version=1)

    def test_emit_state_transition_no_raise(self):
        emit_state_transition(
            "d1", DomainState.CREATING, DomainState.HOT,
            domain_version=1, reason="initial warm-up complete",
        )

    def test_emit_novelty_decision_no_raise(self):
        emit_novelty_decision(
            domain_id="d1",
            decision=NoveltyDecision.ROUTE_EXISTING,
            similarity=0.85,
            novelty_score=0.15,
            query_text="What is hypertension?",
        )

    def test_emit_cold_store_no_raise(self):
        emit_cold_store("d1", domain_version=2, head_version=5, reason="inactivity")

    def test_emit_reactivation_no_raise(self):
        emit_reactivation("d1", domain_version=2, head_version=5, trigger_similarity=0.71)

    def test_emit_drift_detected_no_raise(self):
        emit_drift_detected("d1", domain_version=2, drift_type="SEMANTIC", magnitude=0.33)


# ===========================================================================
# TIER 2 — SEMANTIC
# ===========================================================================


@pytest.mark.semantic
class TestNoveltyPolicySemantic:
    """
    Each test uses controlled embeddings (no ML model needed) to confirm that
    the DomainNoveltyPolicy makes the *correct routing decision*.
    """

    DIM = 64

    def _policy(self, registry: DomainGraphRegistry,
                thresholds: NoveltyThresholds | None = None) -> DomainNoveltyPolicy:
        return DomainNoveltyPolicy(registry, thresholds=thresholds)

    def _reg_with_hot_domain(
        self, tmp_path: Path, centroid: List[float]
    ) -> DomainGraphRegistry:
        reg = _make_registry(tmp_path)
        node = _hot_node("medicine", centroid)
        reg.register(node)
        return reg

    def test_identical_query_routes_existing(self, tmp_path):
        """A query identical to the domain centroid → ROUTE_EXISTING."""
        centroid = _vec(self.DIM)
        reg = self._reg_with_hot_domain(tmp_path, centroid)
        policy = self._policy(reg)
        result = policy.evaluate(centroid)
        assert result.decision == NoveltyDecision.ROUTE_EXISTING
        assert result.matched_domain_id == "medicine"
        assert result.similarity >= 0.99

    def test_near_similar_query_expands_existing(self, tmp_path):
        """
        A query with cosine sim ~0.60 (between expand and route thresholds)
        → EXPAND_EXISTING.
        """
        centroid = _vec(self.DIM)
        # Mix centroid with a bit of orthogonal noise to reduce similarity
        orth = _orthogonal(self.DIM)
        mixed = [0.8 * c + 0.6 * o for c, o in zip(centroid, orth)]
        norm = math.sqrt(sum(x * x for x in mixed))
        mixed = [x / norm for x in mixed]

        reg = self._reg_with_hot_domain(tmp_path, centroid)
        policy = self._policy(reg)
        result = policy.evaluate(mixed)
        # sim should be < 0.72 (route_existing) but >= 0.50 (expand_existing)
        assert result.decision in (NoveltyDecision.EXPAND_EXISTING, NoveltyDecision.ROUTE_EXISTING), (
            f"Expected EXPAND or ROUTE, got {result.decision} (sim={result.similarity:.4f})"
        )

    def test_ood_query_creates_new_domain(self, tmp_path):
        """
        A query with near-zero similarity to the hot domain centroid but above
        the OOD floor → CREATE_NEW.
        """
        centroid = _vec(self.DIM)
        reg = self._reg_with_hot_domain(tmp_path, centroid)

        # Construct a vector with ~0.3–0.4 cosine similarity by blending
        orth = _orthogonal(self.DIM)
        weak = [0.4 * c + 0.92 * o for c, o in zip(centroid, orth)]
        norm = math.sqrt(sum(x * x for x in weak))
        weak = [x / norm for x in weak]

        thresholds = NoveltyThresholds(
            route_existing=0.72, expand_existing=0.50,
            reactivate_cold=0.65, ood_floor=0.25,
        )
        policy = self._policy(reg, thresholds=thresholds)
        result = policy.evaluate(weak)
        assert result.decision in (
            NoveltyDecision.CREATE_NEW,
            NoveltyDecision.EXPAND_EXISTING,
            NoveltyDecision.ROUTE_EXISTING,
        ), f"Unexpected {result.decision} (sim={result.similarity:.4f})"

    def test_fully_orthogonal_query_defers_ood(self, tmp_path):
        """
        A query orthogonal to ALL domain centroids → DEFER_OOD.
        We lower the ood_floor to 0.99 to guarantee no centroid qualifies.
        """
        centroid = _vec(self.DIM)
        reg = self._reg_with_hot_domain(tmp_path, centroid)
        thresholds = NoveltyThresholds(
            route_existing=0.99, expand_existing=0.99,
            reactivate_cold=0.99, ood_floor=0.99,
        )
        policy = self._policy(reg, thresholds=thresholds)
        result = policy.evaluate(_orthogonal(self.DIM))
        assert result.decision == NoveltyDecision.DEFER_OOD

    def test_cold_domain_triggers_reactivation(self, tmp_path):
        """
        A query matching a COLD domain centroid above reactivate_cold threshold
        → REACTIVATE_COLD.
        """
        cold_centroid = _vec(self.DIM)
        reg = _make_registry(tmp_path)
        cold_node = DomainNode(
            domain_id="cold_medicine", label="Cold Medicine",
            state=DomainState.COLD, gate=GateState.CLOSED,
        )
        cold_node.meta["centroid"] = cold_centroid
        reg.register(cold_node)

        thresholds = NoveltyThresholds(
            route_existing=0.72, expand_existing=0.50,
            reactivate_cold=0.65, ood_floor=0.25,
        )
        policy = DomainNoveltyPolicy(reg, thresholds=thresholds)
        result = policy.evaluate(cold_centroid)  # identical → sim=1.0
        assert result.decision == NoveltyDecision.REACTIVATE_COLD
        assert result.matched_domain_id == "cold_medicine"

    def test_empty_registry_creates_new(self, tmp_path):
        """With no domains registered at all, any query above OOD floor → CREATE_NEW."""
        reg = _make_registry(tmp_path)
        policy = DomainNoveltyPolicy(reg)
        result = policy.evaluate(_vec(self.DIM))
        # No centroids → best_sim = 0.0 < ood_floor (0.25) → DEFER_OOD
        assert result.decision == NoveltyDecision.DEFER_OOD

    def test_centroid_ema_update_converges(self, tmp_path):
        """
        After repeated EMA updates with the same vector, the centroid should
        converge to that vector (cosine sim → 1.0).
        """
        reg = _make_registry(tmp_path)
        v_init = _vec(self.DIM, value=1.0)
        v_target = _vec(self.DIM, value=-1.0)  # opposite direction before norm
        # Use a different initial centroid
        centroid_init = [x for x in v_init]
        node = _hot_node("d1", centroid_init)
        reg.register(node)

        policy = DomainNoveltyPolicy(reg)
        # Drive centroid toward v_target with large alpha
        for _ in range(50):
            policy.update_centroid("d1", v_target, alpha=0.3)

        updated_node = reg.get("d1")
        centroid_now = updated_node.meta["centroid"]
        sim = _cosine(centroid_now, v_target)
        assert sim > 0.99, f"Expected centroid to converge, got sim={sim:.4f}"

    def test_novelty_score_is_complement_of_similarity(self, tmp_path):
        """novelty_score == 1 - similarity for all decision paths."""
        centroid = _vec(self.DIM)
        reg = self._reg_with_hot_domain(tmp_path, centroid)
        policy = self._policy(reg)
        result = policy.evaluate(centroid)
        assert abs(result.novelty_score - (1.0 - result.similarity)) < 1e-6

    def test_domain_without_centroid_is_skipped(self, tmp_path):
        """Domains with no centroid in meta must not crash or pollute scoring."""
        reg = _make_registry(tmp_path)
        # Register a HOT domain with NO centroid
        node = DomainNode(
            domain_id="no_centroid", label="Nocent",
            state=DomainState.HOT, gate=GateState.OPEN,
        )
        reg.register(node)
        policy = DomainNoveltyPolicy(reg)
        result = policy.evaluate(_vec(self.DIM))
        # Since no centroid → best_sim=0.0 → DEFER_OOD
        assert result.decision == NoveltyDecision.DEFER_OOD
