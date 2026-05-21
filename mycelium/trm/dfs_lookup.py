"""
Phase D — DFS Lookup
======================
Single-worker depth-first traversal over stored IRGraphs.

Consolidation Notes §D.2 / Phase D design:
    Single-worker DFS is the baseline for Phase D.  Phase F will replace
    the plain Python stack with a multi-worker scheduler, but the search
    logic (scoring, ranking, equivalence resolution) is preserved.

Search strategies supported
----------------------------
    BY_HASH          — exact semantic_hash match
    BY_CANONICAL     — case-insensitive substring on canonical_form
    BY_PREDICATE     — exact predicate_family match
    BY_EQUIVALENCE   — any member of the query node’s equivalence_family
    BY_STATE_RANK    — all graphs at or above a minimum state rank

Ranking (result ordering)
-------------------------
    Candidates are ranked by:
        1. state_rank  (CANONICAL > STABILIZED > CANDIDATE > DRAFT)
        2. observation_count (higher = more reused = more trustworthy)
        3. graph_id (alphabetical — ultimate deterministic tie-breaker)

TRMLookupResult
---------------
    A lightweight dataclass (NOT IRGraph) returned by every search method.
    Fields:
        found          : bool
        candidates     : list[dict]   — each is a summary dict, NOT an IRGraph
        primary        : dict | None  — top-ranked candidate summary
        strategy_used  : str
        query_hash     : str
        explanation    : str

    We return summary dicts (not full IRGraph objects) so callers can
    log / serialise results without needing the full IR stack imported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from .promotion import PromotionPolicy
    _POLICY_AVAILABLE = True
except ImportError:
    PromotionPolicy = None  # type: ignore[assignment]
    _POLICY_AVAILABLE = False


SEARCH_STRATEGIES = [
    "BY_HASH",
    "BY_CANONICAL",
    "BY_PREDICATE",
    "BY_EQUIVALENCE",
    "BY_STATE_RANK",
]


@dataclass
class TRMLookupResult:
    """Result of a TRM DFS lookup.

    Attributes
    ----------
    found : bool
        True if at least one candidate was found.
    candidates : list[dict]
        Ranked list of candidate summary dicts (highest rank first).
    primary : dict or None
        The top-ranked candidate, or None if found is False.
    strategy_used : str
        Which SEARCH_STRATEGIES value was used.
    query_hash : str
        The semantic_hash of the query node (for cache keying).
    explanation : str
        Human-readable summary of the lookup result.
    """

    found: bool = False
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    primary: Optional[Dict[str, Any]] = None
    strategy_used: str = "BY_HASH"
    query_hash: str = ""
    explanation: str = ""


class DFSLookup:
    """Single-worker DFS lookup over a GraphStore.

    Parameters
    ----------
    store : GraphStore
        The graph store to search.

    Usage
    -----
    lookup = DFSLookup(store)
    result = lookup.search(ir_node)           # auto-strategy
    result = lookup.find_equivalent(ir_node)  # equivalence_family search
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._policy = PromotionPolicy() if _POLICY_AVAILABLE else None

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(
        self,
        query_node: Any,
        *,
        strategy: str = "BY_HASH",
        min_state_rank: int = 0,
    ) -> TRMLookupResult:
        """Run a DFS lookup against stored graphs.

        Parameters
        ----------
        query_node : IRNode
            The node whose signature drives the search.
        strategy : str
            One of SEARCH_STRATEGIES.  Default BY_HASH.
        min_state_rank : int
            Filter candidates below this rank (0 = include all).

        Returns
        -------
        TRMLookupResult
            Never raises; returns TRMLookupResult(found=False) on failure.
        """
        try:
            sig = query_node.semantic_signature
        except AttributeError:
            return TRMLookupResult(
                found=False,
                explanation="query_node has no semantic_signature",
                strategy_used=strategy,
            )

        query_hash = sig.semantic_hash
        candidates: List[Any] = []

        # ---- DFS traversal (single-worker plain stack) -----------------
        if strategy == "BY_HASH":
            candidates = self._store.find_by_semantic_hash(query_hash)

        elif strategy == "BY_CANONICAL":
            needle = sig.canonical_form.lower()
            candidates = self._dfs_filter(
                lambda g: any(
                    needle in n.semantic_signature.canonical_form.lower()
                    for n in g.nodes
                    if hasattr(n, "semantic_signature")
                )
            )

        elif strategy == "BY_PREDICATE":
            candidates = self._store.find_by_predicate_family(
                sig.predicate_family
            )

        elif strategy == "BY_EQUIVALENCE":
            return self.find_equivalent(query_node)

        elif strategy == "BY_STATE_RANK":
            candidates = self._dfs_filter(
                lambda g: PromotionPolicy.state_rank(g.state) >= min_state_rank
            )

        else:
            return TRMLookupResult(
                found=False,
                explanation=f"Unknown strategy: {strategy}",
                strategy_used=strategy,
                query_hash=query_hash,
            )

        # ---- Filter by min_state_rank ----------------------------------
        if min_state_rank > 0 and self._policy is not None:
            candidates = [
                g for g in candidates
                if PromotionPolicy.state_rank(g.state) >= min_state_rank
            ]

        return self._build_result(candidates, strategy, query_hash)

    def find_equivalent(
        self, query_node: Any
    ) -> TRMLookupResult:
        """Search for graphs sharing any canonical form in equivalence_family.

        Used when an exact semantic_hash match fails but the query node
        may have been canonicalized under a slightly different phrasing
        (same predicate family, same meaning, different surface form).
        """
        try:
            equiv_family = query_node.semantic_signature.equivalence_family or []
            query_hash = query_node.semantic_signature.semantic_hash
        except AttributeError:
            return TRMLookupResult(
                found=False,
                explanation="query_node missing semantic_signature",
                strategy_used="BY_EQUIVALENCE",
            )

        if not equiv_family:
            return TRMLookupResult(
                found=False,
                explanation="No equivalence_family members to search",
                strategy_used="BY_EQUIVALENCE",
                query_hash=query_hash,
            )

        # DFS: check every stored graph for any node whose canonical_form
        # appears in the query node’s equivalence_family
        equiv_set = set(equiv_family)
        matches: List[Any] = []
        seen_ids: set = set()

        stack = list(self._store.all_graph_ids())  # DFS stack seed
        while stack:
            gid = stack.pop()
            if gid in seen_ids:
                continue
            seen_ids.add(gid)
            g = self._store.get_latest(gid)
            if g is None:
                continue
            for node in g.nodes:
                try:
                    if node.semantic_signature.canonical_form in equiv_set:
                        matches.append(g)
                        break
                except AttributeError:
                    pass

        return self._build_result(matches, "BY_EQUIVALENCE", query_hash)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dfs_filter(self, predicate: Any) -> List[Any]:
        """Return all latest graphs satisfying predicate via DFS."""
        results: List[Any] = []
        stack = list(self._store.all_graph_ids())
        visited: set = set()
        while stack:
            gid = stack.pop()
            if gid in visited:
                continue
            visited.add(gid)
            g = self._store.get_latest(gid)
            if g is None:
                continue
            try:
                if predicate(g):
                    results.append(g)
            except Exception:
                pass
        return results

    def _graph_summary(self, graph: Any) -> Dict[str, Any]:
        """Build a JSON-safe summary dict for a graph (not the full object)."""
        node_sigs = []
        for node in graph.nodes:
            try:
                sig = node.semantic_signature
                node_sigs.append({
                    "node_id": node.id,
                    "semantic_hash": sig.semantic_hash,
                    "canonical_form": sig.canonical_form,
                    "predicate_family": sig.predicate_family,
                    "equivalence_family": list(sig.equivalence_family),
                })
            except AttributeError:
                node_sigs.append({"node_id": getattr(node, "id", "?"), "error": "no_sig"})

        return {
            "graph_id": graph.graph_id,
            "state": graph.state,
            "version": graph.version,
            "state_rank": PromotionPolicy.state_rank(graph.state)
            if self._policy else 0,
            "observation_count": self._store.observation_count(graph.graph_id),
            "confidence": getattr(
                graph.confidence_state, "overall_confidence", 0.0
            ),
            "predicate_families": sorted({
                s["predicate_family"]
                for s in node_sigs
                if "predicate_family" in s
            }),
            "node_signatures": node_sigs,
            "parent_graph_id": graph.parent_graph_id,
            "created_at": graph.created_at,
            "updated_at": graph.updated_at,
        }

    def _rank_key(self, summary: Dict[str, Any]) -> tuple:
        """Sort key: state_rank DESC, observation_count DESC, graph_id ASC."""
        return (
            -summary.get("state_rank", 0),
            -summary.get("observation_count", 0),
            summary.get("graph_id", ""),
        )

    def _build_result(
        self,
        candidates: List[Any],
        strategy: str,
        query_hash: str,
    ) -> TRMLookupResult:
        """Convert raw IRGraph candidates to a ranked TRMLookupResult."""
        if not candidates:
            return TRMLookupResult(
                found=False,
                candidates=[],
                primary=None,
                strategy_used=strategy,
                query_hash=query_hash,
                explanation=f"No match found via {strategy}.",
            )

        summaries = [self._graph_summary(g) for g in candidates]
        summaries.sort(key=self._rank_key)
        primary = summaries[0]

        explanation = (
            f"Found {len(summaries)} candidate(s) via {strategy}. "
            f"Top match: {primary['graph_id']} "
            f"(state={primary['state']}, "
            f"obs={primary['observation_count']})."
        )

        return TRMLookupResult(
            found=True,
            candidates=summaries,
            primary=primary,
            strategy_used=strategy,
            query_hash=query_hash,
            explanation=explanation,
        )
