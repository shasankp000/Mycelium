"""
Phase E — Leverage Propagator
==============================
Walks IRGraph leverage edges and applies epistemic decay when a source
node is contested or directly contradicted.

Consolidation Notes £43-44 (damping model):

    effective_leverage = raw_leverage × DAMPING_FACTOR ^ hop

    Where DAMPING_FACTOR = 0.85 (per-hop).  This produces strictly
    decreasing leverage values at every hop, satisfying £44’s key
    invariant that Phase E propagation halts naturally.

    Node confidence update at hop h:
        new_confidence = current_confidence × effective_leverage

    Halting conditions:
        1. effective_leverage < HALT_THRESHOLD (0.01)
        2. Node is DEPRECATED or ARCHIVED
        3. hop_count > max_hops (safety ceiling, default 10)

PropagationResult:
    Records all affected node IDs and the total decay applied so
    the TRM audit trail (Phase D ContradictionReport) has a full
    lineage of confidence degradation without needing to re-run.

Note on edge model:
    Phase A stores leverage information in LeverageEdge objects
    (§43).  However, LeverageEdge objects are not yet linked into
    IRGraph.edges (that happens in Phase F’s full edge builder).
    Phase E therefore performs BFS over IRGraph.edges using edges
    whose relation_type == ‘SUPPORTS’ or ‘DEPENDS_ON’ as leverage
    proxies, with weight as the raw_leverage value.
    If a LeverageEdge list is explicitly passed in, it takes priority.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Per-hop damping factor (£44)
DAMPING_FACTOR: float = 0.85
# Minimum effective leverage before halting
HALT_THRESHOLD: float = 0.01
# Safety ceiling on propagation depth
DEFAULT_MAX_HOPS: int = 10

# Edge relation types treated as leverage proxies
_LEVERAGE_RELATION_TYPES = {"SUPPORTS", "DEPENDS_ON"}


@dataclass
class PropagationResult:
    """Result of one LeveragePropagator.propagate() call.

    Attributes
    ----------
    source_node_id : str
        The node whose contradiction triggered the propagation.
    affected_nodes : list[dict]
        Each dict: {node_id, hop, old_confidence, new_confidence,
        effective_leverage}.
    total_decay_applied : float
        Sum of (old_confidence - new_confidence) across all affected nodes.
    hops_taken : int
        Maximum hop depth reached before halting.
    halt_reason : str
        Why propagation stopped: THRESHOLD | MAX_HOPS | EXHAUSTED.
    """

    source_node_id: str
    affected_nodes: List[Dict[str, Any]] = field(default_factory=list)
    total_decay_applied: float = 0.0
    hops_taken: int = 0
    halt_reason: str = "EXHAUSTED"


class LeveragePropagator:
    """Applies epistemic decay along leverage edges from a contradicted node.

    Parameters
    ----------
    graph_store : GraphStore, optional
        Used to write add_revision() calls when confidence changes exceed
        a threshold.  If None, confidence changes are computed but not
        persisted (dry-run mode).
    max_hops : int
        Maximum propagation depth (default 10).
    persist_threshold : float
        Minimum confidence change to trigger an add_revision().  Avoids
        creating many micro-revisions for negligible changes.
    """

    def __init__(
        self,
        *,
        graph_store: Optional[Any] = None,
        max_hops: int = DEFAULT_MAX_HOPS,
        persist_threshold: float = 0.05,
    ) -> None:
        self._store = graph_store
        self._max_hops = max_hops
        self._persist_threshold = persist_threshold

    def propagate(
        self,
        source_graph: Any,
        source_node_id: str,
        *,
        initial_severity: float = 1.0,
        leverage_edges: Optional[List[Any]] = None,
    ) -> PropagationResult:
        """Propagate epistemic decay from source_node_id through leverage edges.

        Parameters
        ----------
        source_graph : IRGraph
            The graph containing the contradicted node and its edges.
        source_node_id : str
            The node whose confidence has been impacted.
        initial_severity : float
            The contradiction severity [0,1] that seeds the first hop’s
            confidence update.
        leverage_edges : list[LeverageEdge], optional
            Explicit leverage edges.  If None, falls back to IRGraph.edges
            with SUPPORTS / DEPENDS_ON relation types.

        Returns
        -------
        PropagationResult
        """
        result = PropagationResult(source_node_id=source_node_id)

        # Build node lookup: node_id → IRNode
        node_lookup: Dict[str, Any] = {
            n.id: n for n in source_graph.nodes
        }

        # Build edge map: source_id → list[(target_id, raw_leverage)]
        edge_map = self._build_edge_map(source_graph, leverage_edges)

        # BFS / DFS with hop tracking using explicit stack
        # Stack entries: (node_id, hop, effective_leverage)
        stack = [(source_node_id, 0, initial_severity)]
        visited: set = set()
        max_hop_reached = 0

        while stack:
            node_id, hop, eff_leverage = stack.pop()

            if node_id in visited:
                continue
            visited.add(node_id)

            # --- Halting conditions ---
            if hop > self._max_hops:
                result.halt_reason = "MAX_HOPS"
                break
            if eff_leverage < HALT_THRESHOLD:
                result.halt_reason = "THRESHOLD"
                continue  # don’t break — other branches may still propagate

            max_hop_reached = max(max_hop_reached, hop)

            # --- Apply confidence decay to this node ---
            if hop > 0:  # skip the source node itself
                node = node_lookup.get(node_id)
                if node is not None and node.state not in ("DEPRECATED", "ARCHIVED"):
                    old_conf = node.confidence_state.overall_confidence
                    new_conf = max(0.0, old_conf * eff_leverage)
                    delta = old_conf - new_conf

                    result.affected_nodes.append({
                        "node_id": node_id,
                        "hop": hop,
                        "old_confidence": old_conf,
                        "new_confidence": new_conf,
                        "effective_leverage": eff_leverage,
                    })
                    result.total_decay_applied += delta

                    # Persist if store available and delta is significant
                    if self._store is not None and delta >= self._persist_threshold:
                        try:
                            self._store.add_revision(
                                source_graph.graph_id,
                                new_confidence=new_conf,
                            )
                        except Exception as exc:
                            logger.debug("Propagator: add_revision failed: %s", exc)

            # --- Enqueue neighbours ---
            for target_id, raw_lev in edge_map.get(node_id, []):
                next_leverage = raw_lev * (DAMPING_FACTOR ** (hop + 1))
                if next_leverage >= HALT_THRESHOLD:
                    stack.append((target_id, hop + 1, next_leverage))

        result.hops_taken = max_hop_reached
        if result.halt_reason == "EXHAUSTED":
            result.halt_reason = "EXHAUSTED"
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_edge_map(
        self,
        graph: Any,
        leverage_edges: Optional[List[Any]],
    ) -> Dict[str, List[tuple]]:
        """Build source_id → [(target_id, raw_leverage)] mapping.

        Priority: explicit LeverageEdge list > IRGraph.edges proxy.
        """
        edge_map: Dict[str, List[tuple]] = {}

        if leverage_edges:
            # Explicit LeverageEdge objects (£43 data structure)
            for le in leverage_edges:
                try:
                    src = le.source_node_id
                    tgt = le.target_node_id
                    raw = float(le.raw_leverage)
                    edge_map.setdefault(src, []).append((tgt, raw))
                except AttributeError:
                    pass
            return edge_map

        # Fallback: IRGraph.edges with leverage-proxy relation types
        for edge in graph.edges:
            try:
                if edge.relation_type in _LEVERAGE_RELATION_TYPES:
                    src = edge.source
                    tgt = edge.target
                    raw = float(getattr(edge, "weight", 1.0))
                    edge_map.setdefault(src, []).append((tgt, raw))
            except AttributeError:
                pass

        return edge_map
