"""
Phase D — Promotion Policy
===========================
Enforces the GRAPH_STATES lifecycle defined in Consolidation Notes §33:

    DRAFT       → CANDIDATE   (observation_count >= THRESHOLDS["candidate"])
    CANDIDATE   → STABILIZED  (observation_count >= THRESHOLDS["stabilized"])
    STABILIZED  → CANONICAL   (observation_count >= THRESHOLDS["canonical"])
    Any state   → CONTESTED   (ContradictionEdge with severity >= threshold)
    CONTESTED   → DEPRECATED  (TRM explicit demotion call)
    DEPRECATED  → ARCHIVED    (explicit archive call)

All thresholds live in THRESHOLDS; no magic numbers in logic.

Design note: PromotionPolicy is stateless — it receives a graph and an
observation count and returns the target state.  The GraphStore holds the
counts and calls add_revision() on the result.  This separation keeps
promotion logic testable without needing a full GraphStore.
"""

from __future__ import annotations

from typing import Optional

# All promotion thresholds in one place.
# Counts are cumulative observations (re-matches + re-uses).
THRESHOLDS: dict = {
    "candidate":  3,    # DRAFT → CANDIDATE
    "stabilized": 10,   # CANDIDATE → STABILIZED
    "canonical":  25,   # STABILIZED → CANONICAL
    "contradiction_severity": 0.4,   # minimum severity to mark CONTESTED
}

# Ordered rank for comparison (higher = more stable)
_STATE_RANK: dict = {
    "DRAFT":      0,
    "CANDIDATE":  1,
    "STABILIZED": 2,
    "CANONICAL":  3,
    "CONTESTED":  1,   # contested = demoted but not gone
    "DEPRECATED": 0,
    "ARCHIVED":   -1,
}


class PromotionPolicy:
    """Stateless state-transition evaluator for IRGraph objects.

    Usage
    -----
    policy = PromotionPolicy()

    # Normal promotion path:
    target = policy.evaluate(graph, observation_count=5)
    # target == 'CANDIDATE' if graph was DRAFT

    # Contradiction path:
    target = policy.contest(graph, severity=0.7)
    # target == 'CONTESTED'
    """

    THRESHOLDS = THRESHOLDS  # expose for callers
    STATE_RANK = _STATE_RANK

    # ------------------------------------------------------------------
    # Forward promotion (observation-driven)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        graph: any,
        observation_count: int,
    ) -> Optional[str]:
        """Return the new state if a transition is warranted, else None.

        Only forward transitions (DRAFT→CANDIDATE→STABILIZED→CANONICAL)
        are handled here.  CONTESTED / DEPRECATED / ARCHIVED transitions
        have their own methods.

        Parameters
        ----------
        graph : IRGraph
            The graph whose state is being evaluated.
        observation_count : int
            Current cumulative observation count for this graph_id.

        Returns
        -------
        str or None
            Target state if transition should fire, None if no change.
        """
        current = getattr(graph, "state", "DRAFT")

        # Already at top or in a special state — no forward promotion
        if current in ("CANONICAL", "CONTESTED", "DEPRECATED", "ARCHIVED"):
            return None

        if current == "DRAFT" and observation_count >= THRESHOLDS["candidate"]:
            return "CANDIDATE"
        if current == "CANDIDATE" and observation_count >= THRESHOLDS["stabilized"]:
            return "STABILIZED"
        if current == "STABILIZED" and observation_count >= THRESHOLDS["canonical"]:
            return "CANONICAL"

        return None  # no transition yet

    # ------------------------------------------------------------------
    # Contradiction-driven demotion
    # ------------------------------------------------------------------

    def contest(
        self,
        graph: any,
        severity: float,
    ) -> Optional[str]:
        """Return 'CONTESTED' if severity meets the threshold, else None.

        A CONTESTED graph is NOT immediately deprecated.  It remains
        available for lookup but its ConfidenceState.contradiction_penalty
        is updated by TRMEngine.record_contradiction().

        A subsequent explicit call to deprecate() finalises the demotion.
        """
        if severity >= THRESHOLDS["contradiction_severity"]:
            current = getattr(graph, "state", "DRAFT")
            if current not in ("DEPRECATED", "ARCHIVED"):
                return "CONTESTED"
        return None

    def deprecate(self, graph: any) -> Optional[str]:
        """Return 'DEPRECATED' if the graph is currently CONTESTED, else None."""
        current = getattr(graph, "state", "DRAFT")
        if current == "CONTESTED":
            return "DEPRECATED"
        return None

    def archive(self, graph: any) -> Optional[str]:
        """Return 'ARCHIVED' if the graph is DEPRECATED, else None."""
        current = getattr(graph, "state", "DRAFT")
        if current == "DEPRECATED":
            return "ARCHIVED"
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def state_rank(state: str) -> int:
        """Return the numeric rank of a state (higher = more stable)."""
        return _STATE_RANK.get(state, -1)
