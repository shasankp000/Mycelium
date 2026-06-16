"""
tests/domain_graph/test_domain_graph.py

Unit tests for:
  - mycelium.domain_graph.state  (enums)
  - mycelium.domain_graph.models (DomainNode, DomainEdge, DriftProfile)
  - mycelium.domain_graph.registry (DomainGraphRegistry CRUD + query)
  - mycelium.domain_graph.novelty  (DomainNoveltyPolicy decision logic)
  - mycelium.domain_graph.events   (emit_domain_event contract)
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# State enums
# ---------------------------------------------------------------------------

class TestDomainStateEnum:
    def test_all_states_importable(self):
        from mycelium.domain_graph.state import DomainState
        states = ["CREATING", "HOT", "WARM", "COLD", "REMEMBERING",
                  "DEPRECATED", "ARCHIVED", "FAILED"]
        for s in states:
            assert hasattr(DomainState, s), f"DomainState missing {s}"

    def test_domain_mode_importable(self):
        from mycelium.domain_graph.state import DomainMode
        assert hasattr(DomainMode, "RETRIEVAL_ONLY")
        assert hasattr(DomainMode, "FULL_DOMAIN")

    def test_gate_state_importable(self):
        from mycelium.domain_graph.state import GateState
        for s in ["BOOTSTRAP", "EXPANSION", "STABLE", "THAWING", "RECALIBRATING"]:
            assert hasattr(GateState, s)

    def test_novelty_decision_importable(self):
        from mycelium.domain_graph.state import NoveltyDecision
        for s in ["NEW_DOMAIN", "CHILD_DOMAIN", "PATCH_EXISTING",
                  "PROVISIONAL", "RETRIEVAL_ONLY"]:
            assert hasattr(NoveltyDecision, s)

    def test_domain_state_is_str_enum(self):
        from mycelium.domain_graph.state import DomainState
        assert DomainState.HOT == "HOT"
        assert str(DomainState.COLD) == "COLD"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TestDomainNode:
    def test_default_construction(self):
        from mycelium.domain_graph.models import DomainNode
        node = DomainNode(domain_id="physics", label="Physics")
        assert node.domain_id == "physics"
        assert node.label == "Physics"
        assert node.state == "CREATING"
        assert node.mode == "FULL_DOMAIN"
        assert isinstance(node.semantic_centroid, list)
        assert isinstance(node.parent_domains, list)
        assert isinstance(node.activation_stats, dict)

    def test_drift_profile_default(self):
        from mycelium.domain_graph.models import DomainNode, DriftProfile
        node = DomainNode(domain_id="bio", label="Biology")
        assert isinstance(node.drift_profile, DriftProfile)
        assert node.drift_profile.semantic_drift == 0.0

    def test_custom_fields(self):
        from mycelium.domain_graph.models import DomainNode
        node = DomainNode(
            domain_id="chem",
            label="Chemistry",
            state="HOT",
            head_version="3",
            parent_domain_id="science",
        )
        assert node.state == "HOT"
        assert node.head_version == "3"
        assert node.parent_domain_id == "science"


class TestDomainEdge:
    def test_default_construction(self):
        from mycelium.domain_graph.models import DomainEdge
        edge = DomainEdge(
            src_domain_id="physics",
            dst_domain_id="maths",
            relation_type="semantic_neighbor",
        )
        assert edge.weight == 1.0
        assert isinstance(edge.metadata, dict)

    def test_custom_weight(self):
        from mycelium.domain_graph.models import DomainEdge
        edge = DomainEdge(
            src_domain_id="a", dst_domain_id="b",
            relation_type="child_of", weight=0.75,
        )
        assert edge.weight == 0.75


class TestDriftProfile:
    def test_all_fields_default_zero(self):
        from mycelium.domain_graph.models import DriftProfile
        dp = DriftProfile()
        for field_name in ["semantic_drift", "retrieval_drift",
                           "routing_drift", "confidence_drift", "activation_drift"]:
            assert getattr(dp, field_name) == 0.0

    def test_custom_values(self):
        from mycelium.domain_graph.models import DriftProfile
        dp = DriftProfile(semantic_drift=0.15, confidence_drift=0.32)
        assert dp.semantic_drift == 0.15
        assert dp.confidence_drift == 0.32


# ---------------------------------------------------------------------------
# DomainGraphRegistry
# ---------------------------------------------------------------------------

class TestDomainGraphRegistry:
    @pytest.fixture
    def tmp_graph_path(self, tmp_path):
        return tmp_path / "graph.json"

    @pytest.fixture
    def registry(self, tmp_graph_path):
        from mycelium.domain_graph.registry import DomainGraphRegistry
        return DomainGraphRegistry(graph_path=tmp_graph_path)

    @pytest.fixture
    def sample_node(self):
        from mycelium.domain_graph.models import DomainNode
        return DomainNode(
            domain_id="physics",
            label="Physics",
            state="HOT",
            semantic_centroid=[0.1, 0.2, 0.3],
            creation_timestamp=time.time(),
        )

    # CRUD
    def test_add_and_get_domain(self, registry, sample_node):
        registry.add_domain(sample_node)
        got = registry.get_domain("physics")
        assert got is not None
        assert got.domain_id == "physics"

    def test_get_nonexistent_domain_returns_none(self, registry):
        assert registry.get_domain("nonexistent") is None

    def test_update_domain(self, registry, sample_node):
        registry.add_domain(sample_node)
        registry.update_domain("physics", {"state": "COLD"})
        updated = registry.get_domain("physics")
        assert updated.state == "COLD"

    def test_mark_state(self, registry, sample_node):
        from mycelium.domain_graph.state import DomainState
        registry.add_domain(sample_node)
        registry.mark_state("physics", DomainState.WARM)
        node = registry.get_domain("physics")
        assert node.state in ("WARM", DomainState.WARM)

    def test_add_edge(self, registry):
        from mycelium.domain_graph.models import DomainNode, DomainEdge
        n1 = DomainNode(domain_id="a", label="A")
        n2 = DomainNode(domain_id="b", label="B")
        registry.add_domain(n1)
        registry.add_domain(n2)
        edge = DomainEdge(src_domain_id="a", dst_domain_id="b",
                          relation_type="semantic_neighbor")
        registry.add_edge(edge)  # should not raise

    # Query
    def test_all_hot_domains(self, registry):
        from mycelium.domain_graph.models import DomainNode
        hot = DomainNode(domain_id="hot1", label="H1", state="HOT")
        cold = DomainNode(domain_id="cold1", label="C1", state="COLD")
        registry.add_domain(hot)
        registry.add_domain(cold)
        hot_list = registry.all_hot_domains()
        ids = [n.domain_id for n in hot_list]
        assert "hot1" in ids
        assert "cold1" not in ids

    def test_all_cold_domains(self, registry):
        from mycelium.domain_graph.models import DomainNode
        hot = DomainNode(domain_id="hot2", label="H2", state="HOT")
        cold = DomainNode(domain_id="cold2", label="C2", state="COLD")
        registry.add_domain(hot)
        registry.add_domain(cold)
        cold_list = registry.all_cold_domains()
        ids = [n.domain_id for n in cold_list]
        assert "cold2" in ids
        assert "hot2" not in ids

    def test_nearest_domains_returns_list(self, registry):
        from mycelium.domain_graph.models import DomainNode
        for i in range(5):
            node = DomainNode(
                domain_id=f"d{i}", label=f"D{i}",
                state="HOT",
                semantic_centroid=[float(i) * 0.1, float(i) * 0.2, float(i) * 0.3],
            )
            registry.add_domain(node)
        results = registry.nearest_domains([0.1, 0.2, 0.3], top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_nearest_domains_excludes_states(self, registry):
        from mycelium.domain_graph.models import DomainNode
        from mycelium.domain_graph.state import DomainState
        hot = DomainNode(domain_id="h", label="H", state="HOT",
                         semantic_centroid=[0.5, 0.5, 0.5])
        cold = DomainNode(domain_id="c", label="C", state="COLD",
                          semantic_centroid=[0.5, 0.5, 0.5])
        registry.add_domain(hot)
        registry.add_domain(cold)
        results = registry.nearest_domains(
            [0.5, 0.5, 0.5], top_k=10,
            exclude_states=[DomainState.COLD],
        )
        ids = [n.domain_id for n in results]
        assert "c" not in ids

    # Persistence
    def test_snapshot_creates_file(self, registry, sample_node, tmp_graph_path):
        registry.add_domain(sample_node)
        snap = registry.snapshot()
        assert Path(snap).exists() or tmp_graph_path.exists()

    def test_snapshot_is_valid_json(self, registry, sample_node, tmp_graph_path):
        registry.add_domain(sample_node)
        registry.snapshot()
        if tmp_graph_path.exists():
            data = json.loads(tmp_graph_path.read_text())
            assert "nodes" in data or isinstance(data, dict)

    def test_graph_version_returns_string(self, registry):
        version = registry.graph_version()
        assert isinstance(version, str)

    def test_persistence_roundtrip(self, tmp_graph_path):
        from mycelium.domain_graph.registry import DomainGraphRegistry
        from mycelium.domain_graph.models import DomainNode
        reg1 = DomainGraphRegistry(graph_path=tmp_graph_path)
        reg1.add_domain(DomainNode(domain_id="persist_test", label="Persist"))
        reg1.snapshot()

        reg2 = DomainGraphRegistry(graph_path=tmp_graph_path)
        node = reg2.get_domain("persist_test")
        assert node is not None
        assert node.label == "Persist"


# ---------------------------------------------------------------------------
# DomainNoveltyPolicy
# ---------------------------------------------------------------------------

class TestDomainNoveltyPolicy:
    @pytest.fixture
    def registry_with_nodes(self, tmp_path):
        from mycelium.domain_graph.registry import DomainGraphRegistry
        from mycelium.domain_graph.models import DomainNode
        reg = DomainGraphRegistry(graph_path=tmp_path / "g.json")
        reg.add_domain(DomainNode(
            domain_id="physics",
            label="Physics",
            state="HOT",
            semantic_centroid=[1.0, 0.0, 0.0],
        ))
        reg.add_domain(DomainNode(
            domain_id="maths",
            label="Maths",
            state="HOT",
            semantic_centroid=[0.0, 1.0, 0.0],
        ))
        return reg

    def test_novel_embedding_returns_new_domain(self, registry_with_nodes):
        from mycelium.domain_graph.novelty import DomainNoveltyPolicy
        from mycelium.domain_graph.state import NoveltyDecision
        policy = DomainNoveltyPolicy(
            registry=registry_with_nodes,
            spectral_analyzer=None,
            centroid_distance_threshold=0.10,  # tight threshold
        )
        # Completely orthogonal vector — far from both centroids
        decision, nearest = policy.decide(
            query_embedding=[0.0, 0.0, 1.0],
            spectral_signature=None,
            candidate_samples=[],
        )
        assert decision in (
            NoveltyDecision.NEW_DOMAIN,
            NoveltyDecision.PROVISIONAL,
        )

    def test_similar_embedding_returns_patch_or_child(self, registry_with_nodes):
        from mycelium.domain_graph.novelty import DomainNoveltyPolicy
        from mycelium.domain_graph.state import NoveltyDecision
        policy = DomainNoveltyPolicy(
            registry=registry_with_nodes,
            spectral_analyzer=None,
            centroid_distance_threshold=0.99,  # very loose — everything is close
        )
        decision, nearest = policy.decide(
            query_embedding=[0.99, 0.01, 0.0],
            spectral_signature=None,
            candidate_samples=[],
        )
        assert decision in (
            NoveltyDecision.PATCH_EXISTING,
            NoveltyDecision.CHILD_DOMAIN,
            NoveltyDecision.RETRIEVAL_ONLY,
        )
        assert nearest is not None

    def test_decide_returns_tuple(self, registry_with_nodes):
        from mycelium.domain_graph.novelty import DomainNoveltyPolicy
        policy = DomainNoveltyPolicy(
            registry=registry_with_nodes,
            spectral_analyzer=None,
        )
        result = policy.decide(
            query_embedding=[0.5, 0.5, 0.0],
            spectral_signature=None,
            candidate_samples=[],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_registry_returns_new_domain(self, tmp_path):
        from mycelium.domain_graph.registry import DomainGraphRegistry
        from mycelium.domain_graph.novelty import DomainNoveltyPolicy
        from mycelium.domain_graph.state import NoveltyDecision
        reg = DomainGraphRegistry(graph_path=tmp_path / "empty.json")
        policy = DomainNoveltyPolicy(registry=reg, spectral_analyzer=None)
        decision, nearest = policy.decide(
            query_embedding=[0.1, 0.2, 0.3],
            spectral_signature=None,
            candidate_samples=[],
        )
        assert decision == NoveltyDecision.NEW_DOMAIN
        assert nearest is None


# ---------------------------------------------------------------------------
# Domain events
# ---------------------------------------------------------------------------

class TestDomainEvents:
    def test_emit_domain_event_valid_type(self):
        from mycelium.domain_graph import events as ev
        with mock.patch.object(ev, "emit_event", return_value=None) as m:
            ev.emit_domain_event(
                event_type="graph_domain_create",
                domain_id="physics",
                domain_version="1",
                graph_schema_version="1.0",
                sequence_number=1,
                payload={"reason": "test"},
            )
            assert m.called

    def test_emit_domain_event_invalid_type_raises(self):
        from mycelium.domain_graph import events as ev
        with pytest.raises((AssertionError, ValueError)):
            ev.emit_domain_event(
                event_type="not_a_valid_event",
                domain_id="x",
                domain_version="1",
                graph_schema_version="1.0",
                sequence_number=0,
            )

    def test_all_event_types_are_strings(self):
        from mycelium.domain_graph.events import EVENT_TYPES
        assert all(isinstance(e, str) for e in EVENT_TYPES)
        assert len(EVENT_TYPES) >= 6
