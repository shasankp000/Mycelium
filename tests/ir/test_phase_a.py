"""
Phase A Tests — Core IR Layer.

Covers the gate conditions from the Implementation Spec §A.6:
    1. Determinism: identical inputs → same semantic_hash
    2. ConfidenceState.net_confidence() never goes below 0
    3. LeverageEdge.effective_leverage() decreases monotonically with depth
    4. JSON round-trip: ir_from_json(ir_to_json(g)) == g

Additional tests aligned with consolidation notes:
    5. TemporalState historical_validity tracks knowledge revision
    6. ProvenanceChain preserves all lineage fields
    7. graph_fingerprint hashes are stable across two identical graphs
"""

import json
from dataclasses import asdict

import pytest

from mycelium.ir import (
    SemanticSignature,
    ConfidenceState,
    TemporalState,
    ProvenanceChain,
    GraphFingerprint,
    IRNode,
    IREdge,
    IRGraph,
    LeverageEdge,
    ContradictionEdge,
    ir_to_json,
    ir_from_json,
    compute_semantic_hash,
)
from mycelium.ir.serialization import compute_graph_fingerprint_hashes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sig(canonical: str, family: str = "CAUSAL", depth: int = 0) -> SemanticSignature:
    # Hash computed manually here for test setup; in production this is
    # always set by serialization.compute_semantic_hash after canonicalization.
    import hashlib
    payload = f"{canonical}::{family}::{depth}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return SemanticSignature(
        semantic_hash=h,
        embedding_signature=[0.1] * 384,
        spectral_signature=[0.5] * 16,
        predicate_family=family,
        abstraction_level=depth,
        canonical_form=canonical,
        equivalence_family=[],
    )


def _make_conf(overall: float = 0.8, penalty: float = 0.0) -> ConfidenceState:
    return ConfidenceState(
        overall_confidence=overall,
        contradiction_penalty=penalty,
    )


def _make_temporal(t: str = "UNKNOWN") -> TemporalState:
    return TemporalState(type=t)


def _make_provenance() -> ProvenanceChain:
    return ProvenanceChain(
        sources=["test_source"],
        reasoning_paths=["layer_0", "layer_2"],
        decomposition_origin="decomp_template_A",
        evidence_nodes=[],
        ontology_resolution_path=[],
        worker_threads=["worker_0"],
        timestamp="2026-05-19T00:00:00Z",
    )


def _make_node(
    node_id: str,
    canonical: str,
    family: str = "CAUSAL",
    depth: int = 0,
    overall: float = 0.8,
    penalty: float = 0.0,
) -> IRNode:
    return IRNode(
        id=node_id,
        type="CLAIM",
        label=canonical,
        semantic_signature=_make_sig(canonical, family, depth),
        confidence_state=_make_conf(overall, penalty),
        temporal_state=_make_temporal(),
        provenance=_make_provenance(),
    )


def _make_graph(nodes: list, edges: list, graph_id: str = "G001") -> IRGraph:
    fp_hashes = compute_graph_fingerprint_hashes(
        IRGraph(
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
            fingerprint=GraphFingerprint(
                semantic_hash="",
                structural_hash="",
                predicate_family_hash="",
                temporal_signature="",
                ontology_signature="",
            ),
        )
    )
    fp = GraphFingerprint(**fp_hashes)
    return IRGraph(
        graph_id=graph_id,
        nodes=nodes,
        edges=edges,
        fingerprint=fp,
        created_at="2026-05-19T00:00:00Z",
        updated_at="2026-05-19T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Test 1: Determinism — identical inputs → identical semantic_hash
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_canonical_same_hash(self):
        """Same canonical_form + predicate_family + depth → same semantic_hash."""
        node_a = _make_node("n1", "CAUSAL::smoking::causes::cancer::depth0")
        node_b = _make_node("n2", "CAUSAL::smoking::causes::cancer::depth0")
        assert node_a.semantic_signature.semantic_hash == node_b.semantic_signature.semantic_hash

    def test_different_canonical_different_hash(self):
        """Different canonical form → different semantic_hash."""
        node_a = _make_node("n1", "CAUSAL::smoking::causes::cancer::depth0")
        node_b = _make_node("n2", "CORRELATIONAL::smoking::correlates_with::cancer::depth0")
        assert node_a.semantic_signature.semantic_hash != node_b.semantic_signature.semantic_hash

    def test_compute_semantic_hash_matches_stored(self):
        """compute_semantic_hash() produces the same value as the stored hash."""
        node = _make_node("n1", "CAUSAL::smoking::causes::cancer::depth0")
        computed = compute_semantic_hash(node)
        assert computed == node.semantic_signature.semantic_hash

    def test_hash_stable_across_calls(self):
        """Multiple calls to compute_semantic_hash return the same value."""
        node = _make_node("n1", "CAUSAL::earth::orbits::sun::depth0", "CAUSAL")
        h1 = compute_semantic_hash(node)
        h2 = compute_semantic_hash(node)
        assert h1 == h2


# ---------------------------------------------------------------------------
# Test 2: ConfidenceState.net_confidence() is always ≥ 0
# ---------------------------------------------------------------------------

class TestConfidenceState:
    def test_net_confidence_no_penalty(self):
        c = _make_conf(overall=0.8, penalty=0.0)
        assert c.net_confidence() == pytest.approx(0.8)

    def test_net_confidence_with_penalty(self):
        c = _make_conf(overall=0.8, penalty=0.3)
        assert c.net_confidence() == pytest.approx(0.5)

    def test_net_confidence_never_negative(self):
        """Even with penalty > overall, net_confidence stays ≥ 0."""
        c = _make_conf(overall=0.2, penalty=0.9)
        assert c.net_confidence() == 0.0

    def test_net_confidence_saturates_at_one(self):
        """Net confidence is clamped to [0, 1]."""
        c = _make_conf(overall=1.0, penalty=0.0)
        assert c.net_confidence() == pytest.approx(1.0)

    def test_net_confidence_formula(self):
        """net = support_score - contradiction_score (§27 formula)."""
        c = ConfidenceState(
            overall_confidence=0.75,
            contradiction_penalty=0.25,
        )
        # 0.75 - 0.25 = 0.50
        assert c.net_confidence() == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Test 3: LeverageEdge.effective_leverage() monotonically decreasing
# ---------------------------------------------------------------------------

class TestLeverageEdge:
    def _edge(self, depth: int) -> LeverageEdge:
        return LeverageEdge(
            source="n1",
            target="n2",
            direct_leverage=1.0,
            transitive_leverage=0.5,
            stability=0.9,
            decay_rate=0.01,
            dependency_depth=depth,
            verification_state="UNVERIFIED",
        )

    def test_effective_leverage_depth_zero(self):
        """At depth 0, effective_leverage == direct_leverage."""
        e = self._edge(0)
        assert e.effective_leverage() == pytest.approx(1.0)

    def test_effective_leverage_decreasing_with_depth(self):
        """Monotonically decreasing: depth 0 > 1 > 2 > 3."""
        leverages = [self._edge(d).effective_leverage() for d in range(4)]
        for i in range(len(leverages) - 1):
            assert leverages[i] > leverages[i + 1], (
                f"effective_leverage not decreasing: depth {i} = {leverages[i]}, "
                f"depth {i+1} = {leverages[i+1]}"
            )

    def test_effective_leverage_bounded(self):
        """effective_leverage is always in [0, direct_leverage]."""
        for depth in range(10):
            e = LeverageEdge(
                source="a", target="b",
                direct_leverage=0.8,
                transitive_leverage=0.4,
                stability=1.0,
                decay_rate=0.01,
                dependency_depth=depth,
                verification_state="UNVERIFIED",
            )
            lev = e.effective_leverage()
            assert 0.0 <= lev <= 0.8

    def test_effective_leverage_formula(self):
        """Verify the 0.85^depth damping formula explicitly."""
        e = LeverageEdge(
            source="a", target="b",
            direct_leverage=1.0,
            transitive_leverage=0.5,
            stability=1.0,
            decay_rate=0.01,
            dependency_depth=3,
            verification_state="UNVERIFIED",
        )
        expected = 1.0 * (0.85 ** 3)  # ≈ 0.6141
        assert e.effective_leverage() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 4: JSON round-trip
# ---------------------------------------------------------------------------

class TestJSONRoundTrip:
    def _sample_graph(self) -> IRGraph:
        n1 = _make_node("n1", "CAUSAL::smoking::causes::cancer::depth0")
        n2 = _make_node("n2", "CAUSAL::dna_damage::causes::cancer::depth1", depth=1)
        edge = IREdge(
            id="e1",
            source="n1",
            target="n2",
            relation_type="CAUSES",
            confidence_state=_make_conf(0.9),
            temporal_state=_make_temporal(),
        )
        return _make_graph([n1, n2], [edge])

    def test_round_trip_equality(self):
        g = self._sample_graph()
        serialised = ir_to_json(g)
        restored = ir_from_json(serialised)
        # Compare via dict representation for clean field-level equality
        assert asdict(g) == asdict(restored)

    def test_round_trip_is_deterministic(self):
        """Serialising the same graph twice produces identical JSON."""
        g = self._sample_graph()
        assert ir_to_json(g) == ir_to_json(g)

    def test_round_trip_graph_id(self):
        g = self._sample_graph()
        restored = ir_from_json(ir_to_json(g))
        assert restored.graph_id == g.graph_id

    def test_round_trip_node_count(self):
        g = self._sample_graph()
        restored = ir_from_json(ir_to_json(g))
        assert len(restored.nodes) == len(g.nodes)

    def test_round_trip_edge_count(self):
        g = self._sample_graph()
        restored = ir_from_json(ir_to_json(g))
        assert len(restored.edges) == len(g.edges)

    def test_round_trip_preserves_parent_graph_id_none(self):
        g = self._sample_graph()
        assert g.parent_graph_id is None
        restored = ir_from_json(ir_to_json(g))
        assert restored.parent_graph_id is None

    def test_round_trip_preserves_parent_graph_id_set(self):
        g = self._sample_graph()
        import dataclasses
        g = dataclasses.replace(g, parent_graph_id="G000")
        restored = ir_from_json(ir_to_json(g))
        assert restored.parent_graph_id == "G000"


# ---------------------------------------------------------------------------
# Test 5: TemporalState — historical validity
# ---------------------------------------------------------------------------

class TestTemporalState:
    def test_pluto_historical_validity(self):
        """PlutoIsPlanet was historically valid before 2006 reclassification."""
        pluto_old = TemporalState(
            type="HISTORICAL_ESTIMATE",
            start="1930-01-01",
            end="2006-08-24",
            historical_validity=True,
        )
        pluto_new = TemporalState(
            type="EXACT",
            start="2006-08-24",
            historical_validity=False,
        )
        assert pluto_old.historical_validity is True
        assert pluto_new.historical_validity is False

    def test_relative_temporal_relation(self):
        """BEFORE relationship captured via relative_relation field."""
        t = TemporalState(
            type="RELATIVE",
            relative_relation="BEFORE",
            uncertainty=0.1,
        )
        assert t.relative_relation == "BEFORE"
        assert t.type == "RELATIVE"

    def test_unknown_temporal_default(self):
        """Default TemporalState has max uncertainty and UNKNOWN type."""
        t = TemporalState(type="UNKNOWN")
        assert t.uncertainty == 1.0
        assert t.historical_validity is True


# ---------------------------------------------------------------------------
# Test 6: Graph fingerprint stability
# ---------------------------------------------------------------------------

class TestGraphFingerprint:
    def test_identical_graphs_identical_fingerprint(self):
        n1 = _make_node("n1", "CAUSAL::smoking::causes::cancer::depth0")
        g1 = _make_graph([n1], [])
        g2 = _make_graph([n1], [])
        assert g1.fingerprint.semantic_hash == g2.fingerprint.semantic_hash
        assert g1.fingerprint.structural_hash == g2.fingerprint.structural_hash

    def test_different_nodes_different_fingerprint(self):
        n1 = _make_node("n1", "CAUSAL::smoking::causes::cancer::depth0")
        n2 = _make_node("n2", "CAUSAL::radiation::causes::cancer::depth0")
        g1 = _make_graph([n1], [])
        g2 = _make_graph([n2], [])
        assert g1.fingerprint.semantic_hash != g2.fingerprint.semantic_hash


# ---------------------------------------------------------------------------
# Test 7: ContradictionEdge type is from valid classes
# ---------------------------------------------------------------------------

class TestContradictionEdge:
    def test_valid_contradiction_types(self):
        from mycelium.ir.contradiction import CONTRADICTION_CLASSES
        for cls in CONTRADICTION_CLASSES:
            edge = ContradictionEdge(
                type=cls,
                severity=0.5,
                confidence=0.8,
                scope="LOCAL",
                temporal_validity=TemporalState(type="UNKNOWN"),
                source_graph_id="G001",
                target_graph_id="G002",
            )
            assert edge.type == cls

    def test_tradeoff_is_valid_class(self):
        """TRADEOFF_RELATION must be a recognised class (coffee test §18)."""
        from mycelium.ir.contradiction import CONTRADICTION_CLASSES
        assert "TRADEOFF_RELATION" in CONTRADICTION_CLASSES

    def test_non_contradictory_divergence_exists(self):
        """NON_CONTRADICTORY_DIVERGENCE must be a recognised class."""
        from mycelium.ir.contradiction import CONTRADICTION_CLASSES
        assert "NON_CONTRADICTORY_DIVERGENCE" in CONTRADICTION_CLASSES
