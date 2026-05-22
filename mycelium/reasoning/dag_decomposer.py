"""
Phase D — DAG Decomposer
=========================
DFS-style recursive claim decomposition that builds an IRGraph from a root
IRNode by recursively generating sub-claims.

Key guarantees (Consolidation Notes §D.1):
    1. Cycle safety          — visited set keyed on semantic_hash prevents
                               infinite recursion along identical sub-paths.
    2. DAG reuse             — if a sub-claim semantic_hash already appeared
                               in any previous branch, the existing node
                               reference is returned instead of duplicating it.
    3. Info-gain gating      — a node is expanded only when its confidence
                               uncertainty exceeds `info_gain_threshold`.
    4. Depth bounding        — recursion halts at `max_depth`.
    5. GraphStore / Promotion integration — each produced graph is stored and
                               observed; PromotionPolicy runs after every
                               observe() call so that frequently-derived
                               sub-graphs climb the lifecycle ladder.

Usage
-----
    from mycelium.reasoning.dag_decomposer import DAGDecomposer
    from mycelium.trm.graph_store import GraphStore

    store = GraphStore()
    decomposer = DAGDecomposer(graph_store=store, max_depth=3)
    graph = decomposer.decompose(root_node)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Optional hard dependencies — fail gracefully so tests can mock them
# ---------------------------------------------------------------------------
try:
    from mycelium.ir.graph import IRGraph, IRNode, IREdge
    from mycelium.ir.primitives import (
        ConfidenceState,
        GraphFingerprint,
        SemanticSignature,
        TemporalState,
        ProvenanceChain,
    )
    _IR_AVAILABLE = True
except ImportError:  # pragma: no cover
    IRGraph = None  # type: ignore[assignment,misc]
    IRNode = None  # type: ignore[assignment,misc]
    IREdge = None  # type: ignore[assignment,misc]
    ConfidenceState = None  # type: ignore[assignment,misc]
    GraphFingerprint = None  # type: ignore[assignment,misc]
    SemanticSignature = None  # type: ignore[assignment,misc]
    TemporalState = None  # type: ignore[assignment,misc]
    ProvenanceChain = None  # type: ignore[assignment,misc]
    _IR_AVAILABLE = False

try:
    from mycelium.trm.graph_store import GraphStore
    from mycelium.trm.promotion import PromotionPolicy
    _TRM_AVAILABLE = True
except ImportError:  # pragma: no cover
    GraphStore = None  # type: ignore[assignment,misc]
    PromotionPolicy = None  # type: ignore[assignment,misc]
    _TRM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_semantic_hash(canonical_form: str, predicate_family: str, abstraction_level: int) -> str:
    """SHA-256 of canonical_form + predicate_family + abstraction_level.

    Must be deterministic: same inputs → same hash (spec §A.5).
    """
    raw = f"{canonical_form}|{predicate_family}|{abstraction_level}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _leaf_graph(node: any, graph_id: str) -> any:
    """Return a minimal single-node IRGraph for a leaf (non-expanded) node."""
    if not _IR_AVAILABLE:
        raise RuntimeError("DAGDecomposer: mycelium IR stack unavailable")

    import datetime
    now = datetime.datetime.now(datetime.UTC).isoformat()

    fp = GraphFingerprint(
        semantic_hash=node.semantic_signature.semantic_hash,
        structural_hash=node.semantic_signature.semantic_hash,
        predicate_family_hash=hashlib.sha256(
            node.semantic_signature.predicate_family.encode()
        ).hexdigest(),
        temporal_signature="",
        ontology_signature="",
        canonicalization_version="v1.0",
    )

    return IRGraph(
        graph_id=graph_id,
        nodes=[node],
        edges=[],
        fingerprint=fp,
        state="DRAFT",
        version="v1",
        confidence_state=ConfidenceState(
            overall_confidence=node.confidence_state.overall_confidence
        ),
        ontology_version="1.0",
        created_at=now,
        updated_at=now,
        parent_graph_id=None,
    )


def _merge_subgraphs(root: any, children: List[any], graph_id: str) -> any:
    """Merge root node and all child sub-graphs into a single IRGraph."""
    if not _IR_AVAILABLE:
        raise RuntimeError("DAGDecomposer: mycelium IR stack unavailable")

    import datetime
    now = datetime.datetime.now(datetime.UTC).isoformat()

    all_nodes: List[any] = [root]
    all_edges: List[any] = []
    seen_node_ids: Set[str] = {root.id}

    for child_graph in children:
        for node in child_graph.nodes:
            if node.id not in seen_node_ids:
                all_nodes.append(node)
                seen_node_ids.add(node.id)
        all_edges.extend(child_graph.edges)

        # Add DERIVES_FROM edge from child root node to parent root
        if child_graph.nodes:
            child_root = child_graph.nodes[0]
            edge = IREdge(
                id=f"E-{root.id[:8]}-{child_root.id[:8]}",
                source=root.id,
                target=child_root.id,
                relation_type="DERIVED_FROM",
                confidence_state=ConfidenceState(
                    overall_confidence=min(
                        root.confidence_state.overall_confidence,
                        child_root.confidence_state.overall_confidence,
                    )
                ),
                temporal_state=TemporalState(),
                weight=1.0,
            )
            all_edges.append(edge)

    # Fingerprint over the merged structure
    merged_hash = hashlib.sha256(
        "||".join(sorted(n.semantic_signature.semantic_hash for n in all_nodes)).encode()
    ).hexdigest()

    fp = GraphFingerprint(
        semantic_hash=merged_hash,
        structural_hash=merged_hash,
        predicate_family_hash=hashlib.sha256(
            root.semantic_signature.predicate_family.encode()
        ).hexdigest(),
        temporal_signature="",
        ontology_signature="",
        canonicalization_version="v1.0",
    )

    overall_conf = sum(n.confidence_state.overall_confidence for n in all_nodes) / len(
        all_nodes
    )

    return IRGraph(
        graph_id=graph_id,
        nodes=all_nodes,
        edges=all_edges,
        fingerprint=fp,
        state="DRAFT",
        version="v1",
        confidence_state=ConfidenceState(overall_confidence=overall_conf),
        ontology_version="1.0",
        created_at=now,
        updated_at=now,
        parent_graph_id=None,
    )


def _generate_sub_claims(node: any) -> List[any]:
    """
    Heuristic sub-claim generator (Phase D v1).

    Strategy: lower the abstraction_level by 1 per sub-claim and
    generate two candidate sub-nodes (cause-side and effect-side).
    Phase F upgrade: replace with LLM-driven claim decomposition.
    """
    if not _IR_AVAILABLE:
        return []

    sig = node.semantic_signature
    base_conf = node.confidence_state.overall_confidence
    sub_level = max(0, sig.abstraction_level - 1)

    sub_claims: List[any] = []
    for i, equiv_form in enumerate(sig.equivalence_family[:2]):  # at most 2 sub-claims
        if not equiv_form:
            continue

        sub_canonical = f"{sig.predicate_family}|{equiv_form}|depth{sub_level}"
        sub_hash = _make_semantic_hash(sub_canonical, sig.predicate_family, sub_level)

        sub_sig = SemanticSignature(
            semantic_hash=sub_hash,
            embedding_signature=[],
            spectral_signature=[],
            predicate_family=sig.predicate_family,
            abstraction_level=sub_level,
            canonical_form=sub_canonical,
            equivalence_family=[equiv_form],
        )

        sub_node = IRNode(
            id=f"N-{sub_hash[:12]}-{i}",
            type="CLAIM",
            label=equiv_form,
            semantic_signature=sub_sig,
            confidence_state=ConfidenceState(
                overall_confidence=max(0.0, base_conf - 0.05 * (i + 1))
            ),
            temporal_state=TemporalState(),
            provenance=ProvenanceChain(
                sources=[],
                reasoning_paths=[node.id],
                decomposition_origin="DAGDecomposer",
                evidence_nodes=[],
                ontology_resolution_path=[],
                worker_threads=[],
                timestamp="",
            ),
            metadata={"parent_node_id": node.id},
            ontology_version="1.0",
            state="DRAFT",
        )
        sub_claims.append(sub_node)

    return sub_claims


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DAGDecomposer:
    """DFS claim decomposer that builds versioned IRGraph objects.

    Implements the Phase D §D.1 decomposition guarantee:
      - Cycle detection via `visited` semantic-hash set
      - DAG reuse via `_node_cache` (hash → existing IRNode)
      - Info-gain gating via `should_expand()`
      - Bounded depth via `max_depth`
      - Automatic GraphStore storage + PromotionPolicy evaluation

    Parameters
    ----------
    graph_store : GraphStore, optional
        Where produced sub-graphs are persisted.  When None the decomposer
        operates in-memory-only mode (useful for unit tests).
    max_depth : int
        Maximum DFS recursion depth (default 3, maps to *balanced* mode;
        pass `depth_config["dfs_max_depth"]` from run_workflow for
        mode-aware control).
    info_gain_threshold : float
        Minimum confidence uncertainty (1 − confidence) required to expand
        a node.  Nodes with confidence ≥ (1 − threshold) are leaves.
    """

    def __init__(
        self,
        graph_store: Optional[any] = None,
        max_depth: int = 3,
        info_gain_threshold: float = 0.15,
    ) -> None:
        self._store = graph_store
        self._policy = PromotionPolicy() if _TRM_AVAILABLE else None
        self.max_depth = max_depth
        self.info_gain_threshold = info_gain_threshold

        # Per-decompose-call state — reset in decompose()
        self._visited: Set[str] = set()          # semantic hashes seen this call
        self._node_cache: Dict[str, any] = {}    # hash → IRNode (DAG reuse)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        claim_node: any,
        *,
        request_id: str = "",
    ) -> any:
        """Decompose *claim_node* into a full IRGraph via DFS.

        Each call resets visited / node_cache so repeated calls on the
        same decomposer instance are independent.

        Parameters
        ----------
        claim_node : IRNode
            The root claim to decompose.
        request_id : str
            Optional trace identifier prepended to generated graph IDs.

        Returns
        -------
        IRGraph
            The merged DAG.  Already stored in graph_store (if provided)
            and observed once for PromotionPolicy.
        """
        self._visited = set()
        self._node_cache = {}
        self._request_id = request_id or uuid.uuid4().hex[:8]

        graph = self._dfs(claim_node, current_depth=0)
        return graph

    # ------------------------------------------------------------------
    # DFS core
    # ------------------------------------------------------------------

    def _dfs(self, node: any, current_depth: int) -> any:
        graph_id = f"{self._request_id}-d{current_depth}-{node.id[:8]}"

        # --- Depth limit ---
        if current_depth >= self.max_depth:
            logger.debug("DAGDecomposer: depth limit at %s", node.id)
            return self._store_and_promote(_leaf_graph(node, graph_id), graph_id)

        sem_hash = node.semantic_signature.semantic_hash

        # --- Cycle detection ---
        if sem_hash in self._visited:
            logger.debug("DAGDecomposer: cycle detected for hash %s", sem_hash[:16])
            # Return existing-node reference (DAG reuse guarantee)
            cached = self._node_cache.get(sem_hash, node)
            return self._store_and_promote(_leaf_graph(cached, graph_id + "-reuse"), graph_id + "-reuse")

        self._visited.add(sem_hash)
        self._node_cache[sem_hash] = node

        # --- Info-gain gate ---
        if not self._should_expand(node):
            logger.debug("DAGDecomposer: below info-gain threshold at %s", node.id)
            return self._store_and_promote(_leaf_graph(node, graph_id), graph_id)

        # --- Recursive expansion ---
        sub_claims = _generate_sub_claims(node)
        if not sub_claims:
            return self._store_and_promote(_leaf_graph(node, graph_id), graph_id)

        child_graphs = [self._dfs(sc, current_depth + 1) for sc in sub_claims]
        merged = _merge_subgraphs(node, child_graphs, graph_id)
        return self._store_and_promote(merged, graph_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_expand(self, node: any) -> bool:
        """Return True when uncertainty > info_gain_threshold.

        Uncertainty = 1 − overall_confidence.  High-confidence nodes
        (uncertainty below threshold) are treated as leaves because
        decomposing them would yield no meaningful information gain.
        """
        uncertainty = 1.0 - node.confidence_state.overall_confidence
        return uncertainty > self.info_gain_threshold

    def _store_and_promote(self, graph: any, graph_id: str) -> any:
        """Store graph in GraphStore, observe it, and run PromotionPolicy.

        Falls back silently when GraphStore is not available (test mode).
        """
        if self._store is None or not _TRM_AVAILABLE:
            return graph

        try:
            self._store.put(graph)
            obs_count = self._store.observe(graph_id)

            if self._policy is not None:
                target_state = self._policy.evaluate(graph, obs_count)
                if target_state is not None and target_state != graph.state:
                    logger.debug(
                        "DAGDecomposer: promoting %s %s → %s (obs=%d)",
                        graph_id, graph.state, target_state, obs_count,
                    )
                    self._store.add_revision(graph_id, new_state=target_state)
        except Exception as exc:
            # Non-fatal — decomposition must not crash the main workflow
            logger.warning("DAGDecomposer._store_and_promote: %s", exc)

        return graph
