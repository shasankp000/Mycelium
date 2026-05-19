"""
Phase D — TRMEngine
====================
Orchestrates §46 step 7 (TRM lookup) and the full write path:
    persist → observe → promote → store revised version.

This is the single entry-point that SemanticRouter (Phase C) and future
Phase E workers call.  It keeps GraphStore, DFSLookup, and PromotionPolicy
coordination in one place.

Key responsibilities
--------------------
1. lookup(ir_nodes):
       Runs DFS on the primary IRNode from Phase C output.
       Tries BY_HASH first; falls back to BY_EQUIVALENCE if no exact match.
       Returns TRMLookupResult (never raises).

2. persist(ir_graph):
       Stores the DRAFT graph, increments its observation count, runs
       PromotionPolicy.evaluate(), and calls GraphStore.add_revision()
       if a state transition is warranted.
       Returns the latest (possibly promoted) IRGraph.

3. record_contradiction(graph_id, contradiction_edge):
       Applies a ContradictionEdge to a stored graph:
       - Increments contradiction_penalty on ConfidenceState
       - Calls PromotionPolicy.contest() to check if CONTESTED threshold met
       - Calls add_revision() with new state + updated confidence
       Preserves immutability: original version untouched.

Design notes
------------
    Phase D is single-worker (§D.2 note).  No locking is added here;
    Phase F will wrap TRMEngine in a thread-safe facade.

    TRMEngine holds ONE GraphStore instance.  Callers that need a shared
    store should pass the same TRMEngine instance around (singleton pattern)
    rather than constructing multiple engines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .graph_store import GraphStore
from .dfs_lookup import DFSLookup, TRMLookupResult
from .promotion import PromotionPolicy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from mycelium.ir.primitives import ConfidenceState
    _IR_AVAILABLE = True
except ImportError:
    ConfidenceState = None  # type: ignore[assignment,misc]
    _IR_AVAILABLE = False


class TRMEngine:
    """Orchestrates GraphStore + DFSLookup + PromotionPolicy.

    Parameters
    ----------
    store : GraphStore, optional
        An existing GraphStore.  If None, a fresh one is created.
        Pass a shared instance to give multiple routers the same store.

    Example
    -------
    >>> trm = TRMEngine()
    >>> result = trm.lookup(ir_nodes)            # §46 step 7
    >>> trm.persist(ir_graph)                    # store + auto-promote
    >>> trm.record_contradiction(gid, edge)      # contest + revise
    """

    def __init__(self, store: Optional[GraphStore] = None) -> None:
        self.store = store if store is not None else GraphStore()
        self._dfs = DFSLookup(self.store)
        self._policy = PromotionPolicy()

    # ------------------------------------------------------------------
    # §46 Step 7 — TRM lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        ir_nodes: List[Any],
        *,
        strategy: str = "BY_HASH",
        fallback_to_equivalence: bool = True,
    ) -> TRMLookupResult:
        """Run DFS lookup on the primary IRNode (§46 step 7).

        Parameters
        ----------
        ir_nodes : list[IRNode]
            The list produced by IRBridge / CanonicalizeAndHash.
            Only the first node (primary claim) drives the lookup;
            subsequent nodes are stored but not used as search keys.
        strategy : str
            Initial search strategy (default BY_HASH).
        fallback_to_equivalence : bool
            If the primary strategy returns no matches, retry with
            BY_EQUIVALENCE using the primary node’s equivalence_family.

        Returns
        -------
        TRMLookupResult
            .found == False if store is empty or no match.
        """
        if not ir_nodes:
            return TRMLookupResult(
                found=False,
                explanation="lookup: ir_nodes list is empty",
                strategy_used=strategy,
            )

        primary_node = ir_nodes[0]
        result = self._dfs.search(primary_node, strategy=strategy)

        # Fallback: equivalence family search
        if not result.found and fallback_to_equivalence:
            equiv_result = self._dfs.find_equivalent(primary_node)
            if equiv_result.found:
                logger.debug(
                    "TRMEngine.lookup: BY_HASH miss → BY_EQUIVALENCE hit (%d candidates)",
                    len(equiv_result.candidates),
                )
                return equiv_result

        return result

    # ------------------------------------------------------------------
    # Write path — persist + auto-promote
    # ------------------------------------------------------------------

    def persist(
        self,
        ir_graph: Any,
        *,
        force_observe: bool = True,
    ) -> Any:
        """Store a graph, observe it, and auto-promote if threshold met.

        Steps:
            1. Put graph in store (if not already present by graph_id)
            2. Increment observation count
            3. Evaluate promotion policy
            4. If promotion warranted, call add_revision(new_state=…)

        Parameters
        ----------
        ir_graph : IRGraph
            The DRAFT graph from Phase C IRBridge.build().
        force_observe : bool
            If True, always call observe() even if the graph already
            exists (re-match = new observation).  Default True.

        Returns
        -------
        IRGraph
            The latest (possibly promoted) version of the graph.
        """
        gid = ir_graph.graph_id

        # Check if already stored
        existing = self.store.get_latest(gid)
        if existing is None:
            self.store.put(ir_graph)
            logger.debug("TRMEngine.persist: new graph stored: %s", gid)
        else:
            logger.debug("TRMEngine.persist: graph already known: %s", gid)

        # Increment observation count
        if force_observe:
            obs = self.store.observe(gid)
        else:
            obs = self.store.observation_count(gid)

        # Evaluate promotion
        latest = self.store.get_latest(gid)
        target_state = self._policy.evaluate(latest, obs)
        if target_state is not None:
            latest = self.store.add_revision(
                gid, new_state=target_state
            )
            logger.debug(
                "TRMEngine.persist: promoted %s → %s (obs=%d)",
                gid, target_state, obs,
            )

        return latest

    # ------------------------------------------------------------------
    # Contradiction path
    # ------------------------------------------------------------------

    def record_contradiction(
        self,
        graph_id: str,
        contradiction_edge: Any,
    ) -> Any:
        """Apply a ContradictionEdge to a stored graph.

        Steps:
            1. Read current latest version
            2. Compute new contradiction_penalty =
               old_penalty + (severity * (1 - old_penalty)) [clamped 0-1]
            3. Evaluate PromotionPolicy.contest()
            4. Call add_revision() with updated state + confidence

        The additive-but-bounded formula ensures repeated contradictions
        cause diminishing increments (never exceed 1.0), matching the
        §27 net_confidence() semantics.

        Parameters
        ----------
        graph_id : str
            The graph being contested.
        contradiction_edge : ContradictionEdge
            Provides severity and type information.

        Returns
        -------
        IRGraph
            The revised (CONTESTED or unchanged) graph.
        """
        current = self.store.get_latest(graph_id)
        if current is None:
            raise KeyError(
                f"TRMEngine.record_contradiction: graph_id '{graph_id}' not found"
            )

        severity = float(getattr(contradiction_edge, "severity", 0.0))

        # Compute new penalty: additive-but-bounded (§27 invariant)
        old_penalty = current.confidence_state.contradiction_penalty
        new_penalty = min(1.0, old_penalty + severity * (1.0 - old_penalty))
        new_confidence = max(
            0.0,
            current.confidence_state.overall_confidence - severity,
        )

        # Determine new state
        target_state: Optional[str] = self._policy.contest(current, severity)

        revised = self.store.add_revision(
            graph_id,
            new_state=target_state if target_state else current.state,
            new_confidence=new_confidence,
        )

        logger.debug(
            "TRMEngine.record_contradiction: %s → state=%s penalty=%.3f",
            graph_id,
            revised.state,
            new_penalty,
        )
        return revised

    # ------------------------------------------------------------------
    # Convenience queries (thin pass-through to GraphStore)
    # ------------------------------------------------------------------

    def find_canonical(self) -> List[Any]:
        """Return all CANONICAL graphs (highest-stability tier)."""
        return self.store.find_by_state("CANONICAL")

    def find_contested(self) -> List[Any]:
        """Return all CONTESTED graphs (under contradiction review)."""
        return self.store.find_by_state("CONTESTED")

    def graph_count(self) -> int:
        """Total number of unique graph IDs in the store."""
        return len(self.store)

    def stats(self) -> Dict[str, Any]:
        """Aggregate store statistics for monitoring."""
        store = self.store
        all_ids = store.all_graph_ids()
        state_counts: Dict[str, int] = {}
        for gid in all_ids:
            g = store.get_latest(gid)
            if g:
                state_counts[g.state] = state_counts.get(g.state, 0) + 1
        return {
            "total_graphs": len(all_ids),
            "state_distribution": state_counts,
            "total_observations": sum(
                store.observation_count(gid) for gid in all_ids
            ),
        }
