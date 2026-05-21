"""
Phase A — ContradictionEdge primitive.

Consolidation Notes §32, §18.

Core realization: contradiction is a semantic relation classification
problem, not a binary flag.  The seven contradiction classes cover the
full spectrum from direct logical contradiction to mere divergence.

Critical design invariant (§18 + Phase E regression test):
    "Coffee improves focus" vs "Coffee increases anxiety" must classify
    as TRADEOFF_RELATION, never DIRECT_CONTRADICTION.
    These are not mutually exclusive — one describes cognitive benefit,
    the other describes physiological cost of the same substance.

Classification logic is implemented in Phase E
(mycelium/contradiction/classifier.py).  This file only defines the
data structure and the valid class names.

scope field maps to the geographic/epistemological extent:
    LOCAL  — affects only the immediate claim context
    DOMAIN — propagates through a knowledge domain
    GLOBAL — system-wide ontological contradiction
"""

from __future__ import annotations

from dataclasses import dataclass

from .primitives import TemporalState


CONTRADICTION_CLASSES = [
    "DIRECT_CONTRADICTION",          # same predicate, same scope, logically incompatible
    "PARTIAL_CONTRADICTION",         # same predicate, different scope or degree
    "CONTEXTUAL_CONTRADICTION",      # contradictory only within a specific context
    "TEMPORAL_CONTRADICTION",        # contradictory across different time periods
    "PROBABILISTIC_DISAGREEMENT",    # statistical/probabilistic tension
    "TRADEOFF_RELATION",             # both true but in tension (coffee focus/anxiety)
    "NON_CONTRADICTORY_DIVERGENCE",  # merely different, not contradictory
]


@dataclass
class ContradictionEdge:
    """Typed semantic contradiction between two graphs or claims.

    Parameters
    ----------
    type : str
        One of CONTRADICTION_CLASSES.  Must be assigned by
        ContradictionClassifier in Phase E, never manually guessed.
    severity : float
        Contradiction strength [0, 1].  Used to weight contradiction_penalty
        in ConfidenceState when leverage propagation runs (Phase E).
    confidence : float
        Classifier confidence in the contradiction type assignment.
        Low confidence → classifier should return CONTEXTUAL or
        NON_CONTRADICTORY_DIVERGENCE rather than DIRECT.
    scope : str
        LOCAL | DOMAIN | GLOBAL
    temporal_validity : TemporalState
        When this contradiction is valid.  A TEMPORAL_CONTRADICTION
        between "Pluto is a planet" and "Pluto is not a planet"
        has a TemporalState that captures the 2006 reclassification.
    source_graph_id : str
        graph_id of the first claim.
    target_graph_id : str
        graph_id of the contradicting claim.
    """

    type: str                          # from CONTRADICTION_CLASSES
    severity: float                    # [0, 1]
    confidence: float                  # classifier confidence [0, 1]
    scope: str                         # LOCAL | DOMAIN | GLOBAL
    temporal_validity: TemporalState
    source_graph_id: str
    target_graph_id: str
