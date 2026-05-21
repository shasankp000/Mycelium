"""
Phase A — LeverageEdge primitive.

Consolidation Notes §43-44.

Core realization: support relationships create transitive epistemic leverage.

    A supports B
    B supports C
    ∴ A indirectly supports C

When A is contradicted:
    A leverage on B decays
    → B confidence weakens
    → B leverage on C weakens
    → C destabilises progressively

This creates gradual epistemic decay rather than instant invalidation.

Circular leverage prevention (§44):
    The consolidation notes propose two equivalent formulations:
        effective_leverage = raw_leverage / dependency_depth
    OR
        effective_leverage *= damping_factor (applied per hop)

    We implement the damping_factor model (damping = 0.85 per hop) as it
    produces smoother decay curves and avoids division-by-zero at depth 0.
    The effective_leverage is therefore strictly decreasing with depth,
    which is the key invariant required by Phase E propagation halting.

verification_state lifecycle:
    UNVERIFIED → (truth check) → VERIFIED | CONTESTED
    CONTESTED  → (TRM check)  → VERIFIED | SEVERED
    SEVERED — leverage is permanently removed (§43 Contradiction confirmed)
"""

from __future__ import annotations

from dataclasses import dataclass

# Damping factor per dependency hop — prevents cyclic confidence amplification.
# Derived from §44: effective_leverage = raw_leverage * (damping ^ depth)
_DAMPING_FACTOR: float = 0.85


@dataclass
class LeverageEdge:
    """Represents transitive epistemic support between two IR nodes/graphs.

    Parameters
    ----------
    source : str
        IRNode.id of the supporting claim.
    target : str
        IRNode.id of the dependent claim.
    direct_leverage : float
        Raw support strength [0, 1] at depth 0.
    transitive_leverage : float
        Pre-computed propagated leverage across intermediate nodes.
    stability : float
        Current stability score [0, 1]; decreases under contradiction.
    decay_rate : float
        Domain-specific per-cycle decay rate (Phase F sets this from
        OntologyGovernor.DOMAIN_DECAY_RATES).
    dependency_depth : int
        Number of hops from the original supporting claim.
        Used in effective_leverage() damping calculation.
    verification_state : str
        UNVERIFIED | VERIFIED | CONTESTED | SEVERED
    """

    source: str
    target: str
    direct_leverage: float
    transitive_leverage: float
    stability: float
    decay_rate: float
    dependency_depth: int
    verification_state: str  # UNVERIFIED | VERIFIED | CONTESTED | SEVERED

    def effective_leverage(self) -> float:
        """Compute damped leverage to prevent cyclic confidence amplification.

        Uses the damping factor model from §44:

            effective_leverage = direct_leverage × (damping ^ dependency_depth)

        Properties guaranteed:
          - Monotonically decreasing with dependency_depth
          - Always in [0, direct_leverage]
          - Never amplifies: effective_leverage ≤ direct_leverage

        This is the key invariant that Phase E leverage propagation
        relies upon to guarantee halting when decay drops below 0.01.
        """
        return self.direct_leverage * (_DAMPING_FACTOR ** self.dependency_depth)
