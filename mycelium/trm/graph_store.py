"""
Phase D — Graph Store
======================
Versioned, immutable in-memory store for IRGraph objects.

Design (Consolidation Notes §38-41):
    Immutability contract (§38):
        Stored graphs are NEVER mutated after their first write.
        Every structural change creates a new version via add_revision().
        Version naming: G{id}-v1 → G{id}-v2 → …

    Git-style branching (§39-41):
        branch_graph() forks an existing graph by creating a new IRGraph
        with the original set as parent_graph_id.  Competing hypotheses
        diverge from a common parent rather than overwriting it.

    Observation tracking:
        GraphStore maintains a per-graph_id integer observation_count.
        Each call to observe() increments it.  PromotionPolicy reads
        this counter to decide state transitions.

    Storage layout (in-memory, single process):
        _store: dict[graph_id, list[IRGraph]]   — version history
        _obs:   dict[graph_id, int]             — observation counts
        _hash_index: dict[semantic_hash, set[graph_id]]  — fast hash lookup
"""

from __future__ import annotations

import datetime
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from mycelium.ir.graph import IRGraph
    from mycelium.ir.primitives import ConfidenceState, GraphFingerprint
    _IR_AVAILABLE = True
except ImportError:
    IRGraph = None  # type: ignore[assignment,misc]
    ConfidenceState = None  # type: ignore[assignment,misc]
    GraphFingerprint = None  # type: ignore[assignment,misc]
    _IR_AVAILABLE = False


class GraphStore:
    """Versioned immutable in-memory store for IRGraph objects.

    Thread-safety: not guaranteed (single-worker Phase D).
    Phase F will add a lock wrapper for multi-worker use.
    """

    def __init__(self) -> None:
        # graph_id → ordered list of versions (index 0 = v1)
        self._store: Dict[str, List[any]] = {}
        # graph_id → observation count
        self._obs: Dict[str, int] = {}
        # semantic_hash → set of graph_ids that carry that hash
        self._hash_index: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def put(self, graph: any) -> str:
        """Store a new graph (must not already exist with this graph_id+version).

        Returns the graph_id.
        """
        if not _IR_AVAILABLE:
            raise RuntimeError("GraphStore.put: mycelium IR stack unavailable")

        gid = graph.graph_id
        if gid not in self._store:
            self._store[gid] = []
            self._obs[gid] = 0

        self._store[gid].append(graph)
        self._index_graph(graph)
        logger.debug("GraphStore.put: stored %s (v%d)", gid, len(self._store[gid]))
        return gid

    def observe(self, graph_id: str) -> int:
        """Increment and return the observation count for graph_id.

        Observation = "this graph was seen / reused / matched again".
        PromotionPolicy uses this to decide state promotion.
        """
        if graph_id not in self._obs:
            self._obs[graph_id] = 0
        self._obs[graph_id] += 1
        return self._obs[graph_id]

    def add_revision(
        self,
        graph_id: str,
        *,
        new_state: Optional[str] = None,
        new_confidence: Optional[float] = None,
        extra_metadata: Optional[dict] = None,
    ) -> any:
        """Create a new version of an existing graph (§38 immutability).

        The original graph is untouched.  The new version increments the
        version string (v1 → v2 → …).

        Parameters
        ----------
        graph_id : str
            ID of the graph to revise.
        new_state : str, optional
            Overrides the state field (e.g. DRAFT → CANDIDATE).
        new_confidence : float, optional
            Overrides the overall_confidence in the graph’s ConfidenceState.
        extra_metadata : dict, optional
            Merged into the new graph’s node metadata (not the IRGraph itself
            — IRGraph has no metadata field directly).

        Returns
        -------
        IRGraph
            The new version, already stored.
        """
        current = self.get_latest(graph_id)
        if current is None:
            raise KeyError(f"GraphStore.add_revision: graph_id '{graph_id}' not found")

        version_num = len(self._store[graph_id]) + 1
        new_version_str = f"v{version_num}"
        now = datetime.datetime.utcnow().isoformat() + "Z"

        # Build updated confidence state if requested
        conf = current.confidence_state
        if new_confidence is not None and ConfidenceState is not None:
            conf = ConfidenceState(
                overall_confidence=new_confidence,
                semantic_confidence=conf.semantic_confidence,
                structural_confidence=conf.structural_confidence,
                epistemic_confidence=conf.epistemic_confidence,
                evidence_confidence=conf.evidence_confidence,
                temporal_confidence=conf.temporal_confidence,
                contradiction_penalty=conf.contradiction_penalty,
                aggregation_method=conf.aggregation_method,
                confidence_sources=list(conf.confidence_sources),
            )

        revised = IRGraph(
            graph_id=graph_id,
            nodes=list(current.nodes),       # shallow copy — nodes are immutable
            edges=list(current.edges),
            fingerprint=current.fingerprint,  # fingerprint unchanged by state change
            state=new_state if new_state is not None else current.state,
            version=new_version_str,
            confidence_state=conf,
            ontology_version=current.ontology_version,
            created_at=current.created_at,
            updated_at=now,
            parent_graph_id=current.parent_graph_id,
        )

        self._store[graph_id].append(revised)
        self._index_graph(revised)
        logger.debug(
            "GraphStore.add_revision: %s → %s (state=%s)",
            graph_id, new_version_str, revised.state,
        )
        return revised

    def branch_graph(
        self,
        parent_graph_id: str,
        new_graph_id: str,
        *,
        initial_state: str = "DRAFT",
    ) -> any:
        """Fork a graph (§39-41 branching model).

        Creates a new IRGraph with parent_graph_id set, allowing competing
        hypotheses to diverge from a common ancestor without overwriting it.

        Parameters
        ----------
        parent_graph_id : str
            The graph to fork from.
        new_graph_id : str
            ID for the new branch.
        initial_state : str
            Starting state for the branch (default DRAFT).

        Returns
        -------
        IRGraph
            The new branch graph, already stored.
        """
        parent = self.get_latest(parent_graph_id)
        if parent is None:
            raise KeyError(
                f"GraphStore.branch_graph: parent '{parent_graph_id}' not found"
            )

        now = datetime.datetime.utcnow().isoformat() + "Z"
        branch = IRGraph(
            graph_id=new_graph_id,
            nodes=list(parent.nodes),
            edges=list(parent.edges),
            fingerprint=parent.fingerprint,
            state=initial_state,
            version="v1",
            confidence_state=ConfidenceState(
                overall_confidence=parent.confidence_state.overall_confidence
            ),
            ontology_version=parent.ontology_version,
            created_at=now,
            updated_at=now,
            parent_graph_id=parent_graph_id,   # ← §39 branching link
        )

        self.put(branch)
        logger.debug(
            "GraphStore.branch_graph: %s ← parent=%s",
            new_graph_id, parent_graph_id,
        )
        return branch

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_latest(self, graph_id: str) -> Optional[any]:
        """Return the most recent version of a graph, or None."""
        versions = self._store.get(graph_id)
        return versions[-1] if versions else None

    def get_version(self, graph_id: str, version: str) -> Optional[any]:
        """Return a specific version of a graph, or None."""
        for g in self._store.get(graph_id, []):
            if g.version == version:
                return g
        return None

    def get_all_versions(self, graph_id: str) -> List[any]:
        """Return all versions of a graph, oldest first."""
        return list(self._store.get(graph_id, []))

    def find_by_semantic_hash(
        self, semantic_hash: str
    ) -> List[any]:
        """Return latest versions of all graphs whose nodes include this hash."""
        results: List[any] = []
        for gid in self._hash_index.get(semantic_hash, set()):
            g = self.get_latest(gid)
            if g is not None:
                results.append(g)
        results.sort(key=lambda g: g.graph_id)  # deterministic order
        return results

    def find_by_state(self, state: str) -> List[any]:
        """Return latest versions of all graphs in the given state."""
        results = [
            self.get_latest(gid)
            for gid in self._store
            if self.get_latest(gid) is not None
            and self.get_latest(gid).state == state
        ]
        results.sort(key=lambda g: g.graph_id)
        return results

    def find_by_predicate_family(
        self, predicate_family: str
    ) -> List[any]:
        """Return latest graphs where any node’s predicate_family matches."""
        results: List[any] = []
        for gid in self._store:
            g = self.get_latest(gid)
            if g is None:
                continue
            for node in g.nodes:
                try:
                    if node.semantic_signature.predicate_family == predicate_family:
                        results.append(g)
                        break
                except AttributeError:
                    pass
        results.sort(key=lambda g: g.graph_id)
        return results

    def observation_count(self, graph_id: str) -> int:
        """Return how many times graph_id has been observed."""
        return self._obs.get(graph_id, 0)

    def all_graph_ids(self) -> List[str]:
        """Return all known graph IDs, sorted."""
        return sorted(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index_graph(self, graph: any) -> None:
        """Update the semantic_hash index for fast lookup."""
        try:
            for node in graph.nodes:
                h = node.semantic_signature.semantic_hash
                if h:
                    self._hash_index.setdefault(h, set()).add(graph.graph_id)
        except Exception as exc:
            logger.debug("GraphStore._index_graph: %s", exc)
