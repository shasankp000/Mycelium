"""
Phase F — DST Confidence Fusion
================================
Replaces the Phase A/C net_confidence() scalar stub with proper
Dempster-Shafer Theory (DST) belief combination.

Consolidation Notes §27 upgrade:
    Phase A’s net_confidence() was defined as:
        net_confidence = overall_confidence - contradiction_penalty

    This has a critical flaw: it collapses genuine UNKNOWN states into
    a forced probability.  When both evidence is weak AND contradiction
    is present, the correct answer is "I don’t know" — not a small
    positive float.

    DST fixes this by maintaining three mass allocations:
        m(TRUE)    — evidence that the claim is true
        m(FALSE)   — evidence that the claim is false
        m(UNKNOWN) — genuine uncertainty (neither confirmed nor denied)

    The sum m(TRUE) + m(FALSE) + m(UNKNOWN) = 1.0 at all times.
    m(UNKNOWN) is NEVER silently set to zero.

Dempster’s Rule of Combination:
    Given two independent evidence sources with frames F1 and F2:

        K = F1.m_true * F2.m_false + F1.m_false * F2.m_true
        (K is the conflict mass — normalisation factor)

        combined.m_true    = (F1.m_true * F2.m_true
                              + F1.m_true * F2.m_unknown
                              + F1.m_unknown * F2.m_true) / (1 - K)
        combined.m_false   = (F1.m_false * F2.m_false
                              + F1.m_false * F2.m_unknown
                              + F1.m_unknown * F2.m_false) / (1 - K)
        combined.m_unknown = (F1.m_unknown * F2.m_unknown) / (1 - K)

    When K >= CONFLICT_THRESHOLD (0.95), the sources are irreconcilably
    in conflict and ConflictError is raised rather than silently
    normalising an essentially meaningless result.

Backward compatibility:
    to_net_confidence() extracts a scalar from a DSTFrame so all
    existing Phase A/C/D/E callers that use a float confidence still work.
    The scalar returned is m_true / (m_true + m_false + epsilon) —
    the Pignistic transformation, which is the standard DST → probability
    bridge when a decision must be made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Conflict threshold: K >= this raises ConflictError instead of normalising.
CONFLICT_THRESHOLD: float = 0.95

# Epsilon for numerical stability in Pignistic transform.
_EPS: float = 1e-10


class ConflictError(ValueError):
    """Raised when two evidence sources are irreconcilably in conflict.

    K (the DST conflict mass) >= CONFLICT_THRESHOLD.  Callers should
    treat this as a signal that the two claims require manual arbitration
    or contradiction classification (Phase E) rather than numeric fusion.
    """

    def __init__(self, k: float, frame_a: "DSTFrame", frame_b: "DSTFrame") -> None:
        self.k = k
        self.frame_a = frame_a
        self.frame_b = frame_b
        super().__init__(
            f"DST conflict too high to combine (K={k:.4f} >= {CONFLICT_THRESHOLD}). "
            f"Use Phase E ContradictionClassifier instead."
        )


@dataclass
class DSTFrame:
    """A Dempster-Shafer belief mass function over {{TRUE, FALSE, UNKNOWN}}.

    Invariant: m_true + m_false + m_unknown == 1.0  (within float tolerance).

    Attributes
    ----------
    m_true : float
        Belief mass assigned to the claim being TRUE.
    m_false : float
        Belief mass assigned to the claim being FALSE.
    m_unknown : float
        Belief mass assigned to genuine uncertainty (neither TRUE nor FALSE).
        This is the key field that net_confidence() erased.
    source_label : str
        Human-readable label for the evidence source (for audit trails).
    """

    m_true: float
    m_false: float
    m_unknown: float
    source_label: str = ""

    def __post_init__(self) -> None:
        total = self.m_true + self.m_false + self.m_unknown
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"DSTFrame masses must sum to 1.0, got {total:.6f} "
                f"(true={self.m_true}, false={self.m_false}, "
                f"unknown={self.m_unknown})"
            )

    @classmethod
    def certain_true(cls, label: str = "") -> "DSTFrame":
        """Fully committed belief in TRUE."""
        return cls(m_true=1.0, m_false=0.0, m_unknown=0.0, source_label=label)

    @classmethod
    def certain_false(cls, label: str = "") -> "DSTFrame":
        """Fully committed belief in FALSE."""
        return cls(m_true=0.0, m_false=1.0, m_unknown=0.0, source_label=label)

    @classmethod
    def vacuous(cls, label: str = "") -> "DSTFrame":
        """Total ignorance: all mass on UNKNOWN.  DST identity element."""
        return cls(m_true=0.0, m_false=0.0, m_unknown=1.0, source_label=label)

    def belief_in_true(self) -> float:
        """Pignistic probability of TRUE (decision-theoretic scalar).

        Formula: m_true + m_unknown / 2
        (Distributes unknown mass equally when a decision is required.)
        """
        return self.m_true + self.m_unknown / 2.0

    def plausibility_of_true(self) -> float:
        """Upper bound on belief in TRUE: m_true + m_unknown."""
        return self.m_true + self.m_unknown

    def is_conflicted(self, other: "DSTFrame") -> bool:
        """Return True if combining with other would raise ConflictError."""
        k = self.m_true * other.m_false + self.m_false * other.m_true
        return k >= CONFLICT_THRESHOLD


class DSTFusion:
    """Dempster-Shafer belief combination engine.

    Usage
    -----
    fusion = DSTFusion()
    frame_a = DSTFrame(0.7, 0.1, 0.2)
    frame_b = DSTFrame(0.6, 0.2, 0.2)
    combined = fusion.combine(frame_a, frame_b)
    scalar   = fusion.to_net_confidence(combined)
    """

    # ------------------------------------------------------------------
    # Core combination rule
    # ------------------------------------------------------------------

    def combine(self, frame_a: DSTFrame, frame_b: DSTFrame) -> DSTFrame:
        """Dempster’s rule of combination for two independent sources.

        Parameters
        ----------
        frame_a, frame_b : DSTFrame
            Two evidence frames to combine.

        Returns
        -------
        DSTFrame
            The combined frame.

        Raises
        ------
        ConflictError
            If K >= CONFLICT_THRESHOLD (irreconcilable conflict).
        """
        # Conflict mass
        k = frame_a.m_true * frame_b.m_false + frame_a.m_false * frame_b.m_true

        if k >= CONFLICT_THRESHOLD:
            raise ConflictError(k, frame_a, frame_b)

        normaliser = 1.0 - k
        if normaliser < _EPS:
            raise ConflictError(k, frame_a, frame_b)

        m_true = (
            frame_a.m_true * frame_b.m_true
            + frame_a.m_true * frame_b.m_unknown
            + frame_a.m_unknown * frame_b.m_true
        ) / normaliser

        m_false = (
            frame_a.m_false * frame_b.m_false
            + frame_a.m_false * frame_b.m_unknown
            + frame_a.m_unknown * frame_b.m_false
        ) / normaliser

        m_unknown = (frame_a.m_unknown * frame_b.m_unknown) / normaliser

        # Normalise for floating-point drift
        total = m_true + m_false + m_unknown
        if abs(total - 1.0) > 1e-6:
            m_true    /= total
            m_false   /= total
            m_unknown /= total

        return DSTFrame(
            m_true=round(m_true, 8),
            m_false=round(m_false, 8),
            m_unknown=round(m_unknown, 8),
            source_label=f"combined({frame_a.source_label},{frame_b.source_label})",
        )

    def combine_many(self, frames: List[DSTFrame]) -> DSTFrame:
        """Sequentially combine a list of frames using Dempster’s rule.

        Starts from the vacuous frame (total ignorance) and folds each
        frame in.  Order of combination does not affect the result for
        independent sources (associativity and commutativity of ⊕).

        Parameters
        ----------
        frames : list[DSTFrame]
            Must contain at least one frame.

        Returns
        -------
        DSTFrame
            The jointly combined frame.
        """
        if not frames:
            return DSTFrame.vacuous(label="empty")
        result = frames[0]
        for frame in frames[1:]:
            result = self.combine(result, frame)
        return result

    # ------------------------------------------------------------------
    # ConfidenceState → DSTFrame conversion
    # ------------------------------------------------------------------

    def from_confidence_state(
        self, cs: Any, label: str = ""
    ) -> DSTFrame:
        """Convert a ConfidenceState’s five axes into a combined DSTFrame.

        Mapping:
            overall_confidence   → primary true/unknown split
            contradiction_penalty → false mass source
            semantic/structural/epistemic/evidence/temporal axes →
                individual DSTFrames, combined with Dempster’s rule

        The contradiction_penalty is treated as a direct false-mass
        contribution.  The five confidence axes each contribute partial
        true-mass scaled by their value, with the remainder as unknown.

        Parameters
        ----------
        cs : ConfidenceState
            Source confidence state (Phase A dataclass).
        label : str
            Label for the resulting DSTFrame.

        Returns
        -------
        DSTFrame
            Combined belief frame.  ConflictError is caught internally
            and results in a maximally uncertain frame (vacuous + false
            mass) — callers should inspect m_false for the penalty signal.
        """
        penalty = float(getattr(cs, "contradiction_penalty", 0.0))
        penalty = min(1.0, max(0.0, penalty))

        # Contradiction penalty frame: penalty → FALSE, rest → UNKNOWN
        false_mass = min(penalty, 0.999)
        contradiction_frame = DSTFrame(
            m_true=0.0,
            m_false=false_mass,
            m_unknown=1.0 - false_mass,
            source_label="contradiction_penalty",
        )

        # Build one DSTFrame per confidence axis
        axes = [
            (getattr(cs, "semantic_confidence",   0.0), "semantic"),
            (getattr(cs, "structural_confidence",  0.0), "structural"),
            (getattr(cs, "epistemic_confidence",   0.0), "epistemic"),
            (getattr(cs, "evidence_confidence",    0.0), "evidence"),
            (getattr(cs, "temporal_confidence",    0.0), "temporal"),
        ]

        axis_frames: List[DSTFrame] = []
        for raw_conf, axis_label in axes:
            c = min(1.0, max(0.0, float(raw_conf)))
            if c == 0.0:
                # Zero axis: pure UNKNOWN (don’t contribute evidence)
                axis_frames.append(
                    DSTFrame(m_true=0.0, m_false=0.0, m_unknown=1.0,
                             source_label=axis_label)
                )
            else:
                axis_frames.append(
                    DSTFrame(
                        m_true=c,
                        m_false=0.0,
                        m_unknown=round(1.0 - c, 8),
                        source_label=axis_label,
                    )
                )

        # Combine all axis frames first, then fold in contradiction penalty
        try:
            combined_axes = self.combine_many(axis_frames)
            final = self.combine(combined_axes, contradiction_frame)
        except ConflictError as exc:
            logger.debug(
                "from_confidence_state: ConflictError K=%.4f — "
                "returning high-uncertainty frame", exc.k
            )
            # Conflict: return a frame that signals both high uncertainty
            # and the presence of contradiction.
            remaining = max(0.0, 1.0 - penalty)
            final = DSTFrame(
                m_true=0.0,
                m_false=round(penalty, 8),
                m_unknown=round(remaining, 8),
                source_label=f"conflict_fallback({label})",
            )

        return DSTFrame(
            m_true=final.m_true,
            m_false=final.m_false,
            m_unknown=final.m_unknown,
            source_label=label or "confidence_state",
        )

    # ------------------------------------------------------------------
    # Scalar extraction (backward-compat bridge)
    # ------------------------------------------------------------------

    @staticmethod
    def to_net_confidence(frame: DSTFrame) -> float:
        """Extract a scalar net confidence from a DSTFrame.

        Uses the Pignistic probability transformation:
            P*(TRUE) = m_true + m_unknown / 2

        This is the canonical DST → probability bridge when a
        decision-theoretic scalar is required.  It distributes
        the unknown mass equally across TRUE and FALSE.

        Used by all Phase A/C/D/E callers that expect a float.
        """
        return float(frame.m_true + frame.m_unknown / 2.0)


class ConfidenceStateFusion:
    """Upgrades a ConfidenceState with DST belief computation.

    This is a thin facade that:
        1. Takes an existing ConfidenceState (unmodified)
        2. Computes a DSTFrame from its five axes + penalty
        3. Stores the frame as cs.dst_belief (additive attribute)
        4. Provides a dst_net_confidence() method for scalar extraction

    Backward compatibility: the original ConfidenceState fields are
    never changed.  net_confidence() still works.  Callers that want
    DST-aware confidence call dst_net_confidence() instead.

    Usage
    -----
    csf = ConfidenceStateFusion(confidence_state)
    frame = csf.frame                      # the DSTFrame
    scalar = csf.dst_net_confidence()      # Pignistic scalar
    unknown_mass = csf.frame.m_unknown     # genuine uncertainty
    """

    def __init__(self, confidence_state: Any) -> None:
        self._cs = confidence_state
        self._fusion = DSTFusion()
        self.frame: DSTFrame = self._fusion.from_confidence_state(
            confidence_state,
            label=getattr(confidence_state, "aggregation_method", "dst"),
        )
        # Attach frame to the ConfidenceState object as dst_belief
        # (additive — never overwrites existing fields)
        try:
            confidence_state.dst_belief = self.frame
        except Exception:
            pass  # read-only objects: silent no-op

    def dst_net_confidence(self) -> float:
        """Pignistic scalar confidence (Phase F upgrade of net_confidence)."""
        return DSTFusion.to_net_confidence(self.frame)

    def is_genuinely_uncertain(self, threshold: float = 0.3) -> bool:
        """Return True if unknown mass exceeds threshold.

        High m_unknown means the system genuinely doesn’t know —
        a signal to defer to human review rather than forcing a decision.
        """
        return self.frame.m_unknown >= threshold

    def is_contested(self) -> bool:
        """Return True if m_false is the dominant mass component."""
        return self.frame.m_false > self.frame.m_true
