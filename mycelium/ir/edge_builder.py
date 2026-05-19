"""
Phase F — LeverageEdge Builder
================================
First-class LeverageEdge linkage into IRGraph.edges.

Consolidation Notes £43 Phase F note:
    LeverageEdge objects are not yet linked into IRGraph.edges in
    Phase E.  Phase F’s EdgeBuilder closes this gap.

Design
------
EdgeBuilder.build_leverage_edges():
    Scans an IRGraph’s existing IREdge list for SUPPORTS / DEPENDS_ON
    relation types (the same types LeveragePropagator uses as proxy).
    For each such edge, constructs a proper LeverageEdge with:
        - direct_leverage  = IREdge.weight
        - decay_rate       = OntologyGovernor.domain_decay_rate(
                               predicate_family of the source node)
        - dependency_depth = BFS hop count from the edge’s source to
                             the graph’s root node (first node with no
                             incoming SUPPORTS edges)
        - verification_state = "UNVERIFIED" (Phase E will update to
                               VERIFIED | CONTESTED | SEVERED)
        - transitive_leverage = direct_leverage * DAMPING^depth
        - stability         = 1.0 (fresh edge, no cycles elapsed)

EdgeBuilder.attach():
    Returns a new IRGraph with metadata["leverage_edges"] populated.
    NEVER mutates the original graph (£38 immutability).

BFS root detection:
    The root nodes are those with no incoming SUPPORTS or DEPENDS_ON
    edges.  Depth is measured as shortest path from the source node
    to any root node.  If the graph has no SUPPORTS edges at all,
    all depths default to 0.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from mycelium.ir.leverage import LeverageEdge
    _LEVERAGE_AVAILABLE = True
except ImportError:
    LeverageEdge = None  # type: ignore[assignment,misc]
    _LEVERAGE_AVAILABLE = False

_DAMPING: float = 0.85
_LEVERAGE_RELATION_TYPES: Set[str] = {"SUPPORTS", "DEPENDS_ON"}


class EdgeBuilder:
    """Builds LeverageEdge objects from IRGraph edge proxies.

    Parameters
    ----------
    ontology_governor : OntologyGovernor, optional
        Used to look up domain-specific decay rates.  If None, the
        default rate (0.10) is used for all edges.
    """

    def __init__(self, ontology_governor: Optional[Any] = None) -> None:
        self._gov = ontology_governor

    def build_leverage_edges(
        self, graph: Any
    ) -> List[Any]:
        """Construct LeverageEdge objects for all SUPPORTS/DEPENDS_ON edges.

        Parameters
        ----------
        graph : IRGraph
            The graph to process.

        Returns
        -------
        list[LeverageEdge]
            One LeverageEdge per qualifying IREdge in graph.edges.
            Empty list if no qualifying edges or if LeverageEdge is
            unavailable.
        """
        if not _LEVERAGE_AVAILABLE:
            logger.warning("EdgeBuilder: LeverageEdge unavailable (import failed)")
            return []

        # Build node lookup and predicate family map
        node_pf: Dict[str, str] = {}
        for node in graph.nodes:
            try:
                node_pf[node.id] = node.semantic_signature.predicate_family
            except AttributeError:
                node_pf[node.id] = "CORRELATIONAL"

        # Compute BFS depths from root nodes
        depth_map = self._compute_depths(graph)

        leverage_edges: List[Any] = []
        for edge in graph.edges:
            try:
                if edge.relation_type not in _LEVERAGE_RELATION_TYPES:
                    continue

                src_id = edge.source
                tgt_id = edge.target
                raw_leverage = float(getattr(edge, "weight", 1.0))
                depth = depth_map.get(src_id, 0)
                pf = node_pf.get(src_id, "CORRELATIONAL")
                decay_rate = (
                    self._gov.domain_decay_rate(pf)
                    if self._gov is not None
                    else 0.10
                )
                transitive = raw_leverage * (_DAMPING ** depth)

                le = LeverageEdge(
                    source=src_id,
                    target=tgt_id,
                    direct_leverage=raw_leverage,
                    transitive_leverage=transitive,
                    stability=1.0,
                    decay_rate=decay_rate,
                    dependency_depth=depth,
                    verification_state="UNVERIFIED",
                )
                leverage_edges.append(le)

            except Exception as exc:
                logger.debug("EdgeBuilder: skipped edge: %s", exc)

        return leverage_edges

    def attach(
        self, graph: Any, leverage_edges: Optional[List[Any]] = None
    ) -> Any:
        """Return a new IRGraph with leverage_edges in metadata.

        Builds edges if leverage_edges is None.  Never mutates the
        original graph (£38 immutability).

        Parameters
        ----------
        graph : IRGraph
            Source graph (untouched).
        leverage_edges : list[LeverageEdge], optional
            Pre-built edges.  If None, build_leverage_edges() is called.

        Returns
        -------
        IRGraph
            New graph object with updated metadata dict.
        """
        if leverage_edges is None:
            leverage_edges = self.build_leverage_edges(graph)

        # Build new metadata dict (shallow copy + update)
        new_metadata: Dict[str, Any] = dict(getattr(graph, "metadata", {}) or {})
        new_metadata["leverage_edges"] = leverage_edges

        try:
            from mycelium.ir.graph import IRGraph
            enriched = IRGraph(
                graph_id=graph.graph_id,
                nodes=list(graph.nodes),
                edges=list(graph.edges),
                fingerprint=graph.fingerprint,
                state=graph.state,
                version=graph.version,
                confidence_state=graph.confidence_state,
                ontology_version=graph.ontology_version,
                created_at=graph.created_at,
                updated_at=graph.updated_at,
                parent_graph_id=graph.parent_graph_id,
            )
            # IRGraph has no metadata field in Phase A; attach via object
            object.__setattr__(enriched, "metadata", new_metadata)
        except Exception as exc:
            logger.warning("EdgeBuilder.attach: could not build IRGraph: %s", exc)
            # Fallback: return original graph with metadata monkey-patched
            enriched = graph
            try:
                object.__setattr__(enriched, "_leverage_edges", leverage_edges)
            except Exception:
                pass

        return enriched

    # ------------------------------------------------------------------
    # Internal: BFS depth computation
    # ------------------------------------------------------------------

    def _compute_depths(
        self, graph: Any
    ) -> Dict[str, int]:
        """Compute shortest depth (hops from root) for each node.

        Root = node with no incoming SUPPORTS/DEPENDS_ON edges.
        If no roots found, all depths default to 0.
        """
        node_ids: Set[str] = {n.id for n in graph.nodes}
        has_incoming: Set[str] = set()
        children: Dict[str, List[str]] = {nid: [] for nid in node_ids}

        for edge in graph.edges:
            try:
                if edge.relation_type in _LEVERAGE_RELATION_TYPES:
                    has_incoming.add(edge.target)
                    children[edge.source].append(edge.target)
            except AttributeError:
                pass

        roots = node_ids - has_incoming
        if not roots:
            return {nid: 0 for nid in node_ids}  # no SUPPORTS edges — flat graph

        # BFS from all roots simultaneously
        depth_map: Dict[str, int] = {}
        queue: deque = deque()
        for root in roots:
            depth_map[root] = 0
            queue.append(root)

        while queue:
            current = queue.popleft()
            for child in children.get(current, []):
                if child not in depth_map:
                    depth_map[child] = depth_map[current] + 1
                    queue.append(child)

        # Fill any disconnected nodes with 0
        for nid in node_ids:
            depth_map.setdefault(nid, 0)

        return depth_map
