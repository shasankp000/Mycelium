"""
Phase A — IR Serialization and Semantic Hashing.

Consolidation Notes §22, §36, §46.

Three hard requirements:
  1. Determinism: identical inputs → identical semantic_hash
     (required for cache consistency, graph deduplication, TRM)
  2. Round-trip fidelity: ir_from_json(ir_to_json(g)) == g
  3. Version-awareness: graphs carry canonicalization_version so that
     migrations in Phase D can replay them correctly

Semantic hash algorithm (A.5 of impl spec):
    SHA-256 of: canonical_form + "::" + predicate_family + "::" + str(abstraction_level)

    This is computed AFTER Phase B canonicalization (§46 critical rule:
    semantic_hash generation occurs after canonicalization, before TRM
    persistence).  Never compute hash on raw natural language input.

JSON encoding strategy:
    dataclasses.asdict() produces a nested dict.  Custom encode/decode
    handles re-instantiation of dataclass types on deserialisation,
    preserving Optional[str] fields as None (not as "null" strings).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .primitives import (
    SemanticSignature,
    ConfidenceState,
    TemporalState,
    ProvenanceChain,
    GraphFingerprint,
)
from .graph import IRNode, IREdge, IRGraph
from .leverage import LeverageEdge
from .contradiction import ContradictionEdge


# ---------------------------------------------------------------------------
# Semantic hash
# ---------------------------------------------------------------------------

def compute_semantic_hash(node: IRNode) -> str:
    """Deterministic SHA-256 hash from a node's canonical semantic identity.

    Input string: canonical_form + "::" + predicate_family + "::" + abstraction_level

    Called by Phase B canonicalization pipeline AFTER canonical_form is
    generated, BEFORE TRM persistence (§46 ordering rule).

    The hash is stable across Python versions because we encode to UTF-8
    bytes explicitly and use hashlib rather than Python's built-in hash().
    """
    sig = node.semantic_signature
    payload = (
        f"{sig.canonical_form}"
        f"::{sig.predicate_family}"
        f"::{sig.abstraction_level}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_graph_fingerprint_hashes(graph: IRGraph) -> dict:
    """Compute all four fingerprint hashes for a graph.

    structural_hash    — SHA-256 of sorted node ids + sorted edge (src, tgt, rel) tuples
    predicate_family_hash — SHA-256 of sorted predicate families across nodes
    temporal_signature — SHA-256 of sorted temporal type strings across nodes
    ontology_signature — SHA-256 of graph.ontology_version + all node ontology_versions

    Returns a dict matching GraphFingerprint field names (excluding canonicalization_version).
    """
    node_ids = sorted(n.id for n in graph.nodes)
    edge_tuples = sorted(
        (e.source, e.target, e.relation_type) for e in graph.edges
    )
    predicate_families = sorted(
        n.semantic_signature.predicate_family for n in graph.nodes
    )
    temporal_types = sorted(n.temporal_state.type for n in graph.nodes)
    ontology_parts = [graph.ontology_version] + sorted(
        n.ontology_version for n in graph.nodes
    )

    # semantic_hash of graph = hash of all node semantic_hashes (sorted)
    node_sem_hashes = sorted(n.semantic_signature.semantic_hash for n in graph.nodes)
    semantic_hash = hashlib.sha256(
        "|".join(node_sem_hashes).encode("utf-8")
    ).hexdigest()

    structural_hash = hashlib.sha256(
        json.dumps({"nodes": node_ids, "edges": edge_tuples}).encode("utf-8")
    ).hexdigest()

    predicate_family_hash = hashlib.sha256(
        "|".join(predicate_families).encode("utf-8")
    ).hexdigest()

    temporal_signature = hashlib.sha256(
        "|".join(temporal_types).encode("utf-8")
    ).hexdigest()

    ontology_signature = hashlib.sha256(
        "|".join(ontology_parts).encode("utf-8")
    ).hexdigest()

    return {
        "semantic_hash": semantic_hash,
        "structural_hash": structural_hash,
        "predicate_family_hash": predicate_family_hash,
        "temporal_signature": temporal_signature,
        "ontology_signature": ontology_signature,
    }


# ---------------------------------------------------------------------------
# JSON serialisation — dataclass → dict → JSON string
# ---------------------------------------------------------------------------

def ir_to_json(graph: IRGraph) -> str:
    """Serialise an IRGraph to a JSON string.

    Uses dataclasses.asdict() which recursively converts nested dataclasses.
    The output is deterministic for a given graph (sorted keys).
    """
    return json.dumps(asdict(graph), sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# JSON deserialisation — JSON string → dict → dataclass tree
# ---------------------------------------------------------------------------

def _temporal(d: dict) -> TemporalState:
    return TemporalState(
        type=d["type"],
        start=d.get("start"),
        end=d.get("end"),
        relative_relation=d.get("relative_relation"),
        uncertainty=d.get("uncertainty", 1.0),
        historical_validity=d.get("historical_validity", True),
    )


def _confidence(d: dict) -> ConfidenceState:
    return ConfidenceState(
        overall_confidence=d["overall_confidence"],
        semantic_confidence=d.get("semantic_confidence", 0.0),
        structural_confidence=d.get("structural_confidence", 0.0),
        epistemic_confidence=d.get("epistemic_confidence", 0.0),
        evidence_confidence=d.get("evidence_confidence", 0.0),
        temporal_confidence=d.get("temporal_confidence", 0.0),
        contradiction_penalty=d.get("contradiction_penalty", 0.0),
        aggregation_method=d.get("aggregation_method", "weighted_average"),
        confidence_sources=d.get("confidence_sources", []),
    )


def _provenance(d: dict) -> ProvenanceChain:
    return ProvenanceChain(
        sources=d.get("sources", []),
        reasoning_paths=d.get("reasoning_paths", []),
        decomposition_origin=d.get("decomposition_origin", ""),
        evidence_nodes=d.get("evidence_nodes", []),
        ontology_resolution_path=d.get("ontology_resolution_path", []),
        worker_threads=d.get("worker_threads", []),
        timestamp=d.get("timestamp", ""),
    )


def _semantic_signature(d: dict) -> SemanticSignature:
    return SemanticSignature(
        semantic_hash=d["semantic_hash"],
        embedding_signature=d.get("embedding_signature", []),
        spectral_signature=d.get("spectral_signature", []),
        predicate_family=d.get("predicate_family", "UNKNOWN"),
        abstraction_level=d.get("abstraction_level", 0),
        canonical_form=d.get("canonical_form", ""),
        equivalence_family=d.get("equivalence_family", []),
    )


def _fingerprint(d: dict) -> GraphFingerprint:
    return GraphFingerprint(
        semantic_hash=d["semantic_hash"],
        structural_hash=d["structural_hash"],
        predicate_family_hash=d["predicate_family_hash"],
        temporal_signature=d["temporal_signature"],
        ontology_signature=d["ontology_signature"],
        canonicalization_version=d.get("canonicalization_version", "v1.0"),
    )


def _node(d: dict) -> IRNode:
    return IRNode(
        id=d["id"],
        type=d["type"],
        label=d["label"],
        semantic_signature=_semantic_signature(d["semantic_signature"]),
        confidence_state=_confidence(d["confidence_state"]),
        temporal_state=_temporal(d["temporal_state"]),
        provenance=_provenance(d["provenance"]),
        metadata=d.get("metadata", {}),
        ontology_version=d.get("ontology_version", "v1.0"),
        state=d.get("state", "DRAFT"),
    )


def _edge(d: dict) -> IREdge:
    return IREdge(
        id=d["id"],
        source=d["source"],
        target=d["target"],
        relation_type=d["relation_type"],
        confidence_state=_confidence(d["confidence_state"]),
        temporal_state=_temporal(d["temporal_state"]),
        weight=d.get("weight", 1.0),
        metadata=d.get("metadata", {}),
    )


def ir_from_json(data: str) -> IRGraph:
    """Deserialise a JSON string back into a full IRGraph dataclass tree.

    Round-trip guarantee: ir_from_json(ir_to_json(g)) == g
    (assuming g was produced with the same Python version and field order).
    """
    raw: dict[str, Any] = json.loads(data)
    nodes = [_node(n) for n in raw.get("nodes", [])]
    edges = [_edge(e) for e in raw.get("edges", [])]
    return IRGraph(
        graph_id=raw["graph_id"],
        nodes=nodes,
        edges=edges,
        fingerprint=_fingerprint(raw["fingerprint"]),
        state=raw.get("state", "DRAFT"),
        version=raw.get("version", "v1"),
        confidence_state=_confidence(raw["confidence_state"]),
        ontology_version=raw.get("ontology_version", "v1.0"),
        created_at=raw.get("created_at", ""),
        updated_at=raw.get("updated_at", ""),
        parent_graph_id=raw.get("parent_graph_id"),
    )
