"""
GraphDecayManager
=================
Implements domain-based temporal decay for stored IRGraphs.

Philosophy (Consolidation Notes §6, §20; impl-spec Phase F)
------------------------------------------------------------
Claims and evidence are not eternally valid.  A reasoning graph
stabilized at time T may be incorrect at time T+Δ if:
    - The domain it describes changes (politics, social trends)
    - New evidence contradicts it (medicine, science)
    - Its predicate family had time-bound semantics (temporal claims)

Without decay:
    - Old graphs accumulate and pollute EvidenceGrounder's Tier 1 pool
    - L5 / HypothesisEvaluator draws support from stale evidence
    - Semantic fossilization occurs (the spec explicitly names this)

With decay:
    - Graphs lose confidence over time at domain-appropriate rates
    - When overall_confidence drops below DEPRECATION_FLOOR the graph
      is transitioned to state=DEPRECATED (never deleted, immutability)
    - DEPRECATED graphs are excluded from list_all() by default
    - Any graph can be re-stabilized if fresh reasoning re-observes the
      same semantic hash (TRMStabilizer resets the confidence)

Domain decay rates (from impl-spec Phase F, DOMAIN_DECAY_RATES)
---------------------------------------------------------------
    physics           0.002   / day   (very slow)
    mathematics       0.001   / day   (very slow)
    medicine          0.010   / day   (medium)
    social_trends     0.050   / day   (fast)
    politics          0.040   / day   (fast)
    historical_facts  0.0005  / day   (extremely slow)
    default           0.020   / day   (moderate fallback)

Decay formula (applied per elapsed day since last verification)
---------------------------------------------------------------
    new_confidence = old_confidence * (1 - decay_rate) ^ elapsed_days

    This is exponential decay, not linear, so confidence never hits
    exactly zero — it asymptotically approaches it.  The deprecation
    floor (default 0.10) is where we call the graph semantically dead.

Integration points
------------------
    GraphStore.__init__  → runs _boot_decay_pass() after loading
    GraphStore.put()     → stamps updated_at so next decay is anchored
    idle task / shutdown → can call decay_manager.decay_pass(store)
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from mycelium.trm.graph_store import GraphStore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Domain decay rates (impl-spec Phase F)
# ---------------------------------------------------------------------------

DOMAIN_DECAY_RATES: Dict[str, float] = {
    "physics":          0.002,
    "mathematics":      0.001,
    "medicine":         0.010,
    "medical":          0.010,   # alias
    "social_trends":    0.050,
    "politics":         0.040,
    "historical_facts": 0.0005,
    "history":          0.0005,  # alias
    "default":          0.020,
}

# Predicate-family overrides: some families decay faster than domains alone
PREDICATE_FAMILY_DECAY_MULTIPLIERS: Dict[str, float] = {
    "TEMPORAL":       2.0,   # time-bound claims go stale fastest
    "CORRELATIONAL":  1.5,   # statistical associations shift
    "CAUSAL":         1.0,   # causal claims relatively stable
    "DEFINITIONAL":   0.5,   # definitions change slowly
    "CONTRADICTORY":  1.2,   # contradiction assessments may be revisited
}

# Graph state transitions
STATE_DEPRECATED   = "DEPRECATED"
DEPRECATION_FLOOR  = 0.10   # below this confidence → DEPRECATED


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GraphDecayManager:
    """
    Applies temporal decay to all graphs in a GraphStore.

    Parameters
    ----------
    deprecation_floor : float
        overall_confidence below which a graph is DEPRECATED (default 0.10).
    default_decay_rate : float
        Fallback if domain is unknown (default 0.020 / day).
    """

    def __init__(
        self,
        deprecation_floor: float = DEPRECATION_FLOOR,
        default_decay_rate: float = 0.020,
    ) -> None:
        self._floor = deprecation_floor
        self._default_rate = default_decay_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decay_pass(self, store: "GraphStore") -> Dict[str, int]:
        """
        Run a full decay pass over all graphs in *store*.

        For each graph:
            1. Compute elapsed days since updated_at
            2. Determine domain decay rate from node predicate families
            3. Apply exponential decay to overall_confidence
            4. If new_confidence < deprecation_floor → DEPRECATED revision
            5. Else if confidence changed significantly (≥0.01) → new revision

        Returns a summary dict:
            decayed      : int  — graphs that lost confidence
            deprecated   : int  — graphs newly transitioned to DEPRECATED
            skipped      : int  — already DEPRECATED or fresh (<1 day old)
        """
        summary = {"decayed": 0, "deprecated": 0, "skipped": 0}
        now = datetime.datetime.now(datetime.UTC)

        for gid in store.all_graph_ids():
            graph = store.get_latest(gid)
            if graph is None:
                continue

            # Already deprecated — nothing to do
            if graph.state == STATE_DEPRECATED:
                summary["skipped"] += 1
                continue

            # Historical validity check: if the graph's root node
            # has historical_validity=False, deprecate immediately
            if self._is_historically_invalid(graph):
                self._deprecate(store, graph)
                summary["deprecated"] += 1
                continue

            # Compute elapsed time
            elapsed_days = self._elapsed_days(graph.updated_at, now)
            if elapsed_days < 1.0:
                summary["skipped"] += 1
                continue

            # Determine effective decay rate
            domain    = self._extract_domain(graph)
            family    = self._extract_predicate_family(graph)
            base_rate = DOMAIN_DECAY_RATES.get(domain, self._default_rate)
            multiplier = PREDICATE_FAMILY_DECAY_MULTIPLIERS.get(family, 1.0)
            rate = min(0.999, base_rate * multiplier)   # cap at 99.9% / day

            # Exponential decay
            old_conf = graph.confidence_state.overall_confidence
            new_conf = old_conf * ((1.0 - rate) ** elapsed_days)
            new_conf = max(0.0, new_conf)

            delta = old_conf - new_conf
            if delta < 0.01:
                # Negligible decay — not worth a new revision
                summary["skipped"] += 1
                continue

            logger.debug(
                "GraphDecayManager: %s domain=%s family=%s "
                "conf %.3f → %.3f (elapsed=%.1f days)",
                gid, domain, family, old_conf, new_conf, elapsed_days,
            )

            if new_conf < self._floor:
                self._deprecate(store, graph)
                summary["deprecated"] += 1
            else:
                try:
                    store.add_revision(gid, new_confidence=new_conf)
                    summary["decayed"] += 1
                except Exception as exc:
                    logger.warning(
                        "GraphDecayManager: revision failed for %s (%s)", gid, exc
                    )

        logger.info(
            "GraphDecayManager.decay_pass: decayed=%d deprecated=%d skipped=%d",
            summary["decayed"], summary["deprecated"], summary["skipped"],
        )
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_historically_invalid(graph) -> bool:
        """
        Return True if any node in the graph has
        temporal_state.historical_validity == False.

        A graph that carries an explicitly invalidated claim should be
        deprecated regardless of its confidence score.
        """
        try:
            for node in graph.nodes:
                ts = getattr(node, "temporal_state", None)
                if ts is not None and ts.historical_validity is False:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _elapsed_days(updated_at_iso: str, now: datetime.datetime) -> float:
        """Parse an ISO 8601 timestamp and return elapsed days."""
        try:
            updated = datetime.datetime.fromisoformat(updated_at_iso)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=datetime.UTC)
            delta = now - updated
            return max(0.0, delta.total_seconds() / 86400.0)
        except Exception:
            return 0.0

    @staticmethod
    def _extract_domain(graph) -> str:
        """Best-effort domain extraction from graph nodes."""
        try:
            for node in graph.nodes:
                prov = getattr(node, "provenance", None)
                if prov is None:
                    continue
                for src in getattr(prov, "sources", []):
                    for domain in DOMAIN_DECAY_RATES:
                        if domain in str(src).lower():
                            return domain
        except Exception:
            pass
        return "default"

    @staticmethod
    def _extract_predicate_family(graph) -> str:
        """Return the predicate family of the first node that has one."""
        try:
            for node in graph.nodes:
                sig = getattr(node, "semantic_signature", None)
                if sig is not None:
                    fam = getattr(sig, "predicate_family", "")
                    if fam:
                        return fam
        except Exception:
            pass
        return ""

    @staticmethod
    def _deprecate(store: "GraphStore", graph) -> None:
        """Transition a graph to DEPRECATED (never delete — immutability)."""
        try:
            store.add_revision(
                graph.graph_id,
                new_state=STATE_DEPRECATED,
                new_confidence=max(
                    0.0, graph.confidence_state.overall_confidence * 0.01
                ),
            )
            logger.info(
                "GraphDecayManager: deprecated graph %s", graph.graph_id
            )
        except Exception as exc:
            logger.warning(
                "GraphDecayManager: deprecation failed for %s (%s)",
                graph.graph_id, exc,
            )
