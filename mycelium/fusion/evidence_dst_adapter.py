"""
mycelium/fusion/evidence_dst_adapter.py
========================================
Phase D Step 10 — DSTFusion adapter for structured EvidenceBundle input.

Spec ref: implementation spec v0.2.1 — Phase D, §7 DSTFusion Integration.

Purpose
-------
Previously DSTFusion consumed raw ConfidenceState scalars (one float per
confidence axis).  Phase D upgrades the fusion input to structured semantic
evidence: each falsifiable PredicateFrame now produces a ScoredBundle with
explicit confirmation_score, refutation_score, net_confidence, and an
optional modal_confidence_ceiling.

This module converts that structured evidence into DSTFrames and feeds them
through DSTFusion.combine_many(), producing a single EvidenceDSTResult that
the rest of the pipeline can consume as a first-class belief state.

Key design rules (from spec §7)
--------------------------------
1. DSTFusion now operates over *structured semantic evidence*, NOT raw text.
2. MODAL ceiling is surfaced as an explicit annotation — never a silent
   score reduction.  The ceiling is stored in modal_ceiling_applied and
   emitted on the SSE stream so the UI can label it correctly.
3. DSTFusion must NOT penalise the expert system for low confidence on
   MODAL claims — the ceiling is an epistemic constraint on the original
   claim, not evidence of weak retrieval.
4. Backward compatibility: adapt_expert_outputs() injects 'evidence_dst'
   into the existing expert_outputs dict without touching any other key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mycelium.fusion.dst_fusion import (
    ConflictError,
    DSTFrame,
    DSTFusion,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Threshold below which m_unknown is considered "genuinely uncertain".
_UNCERTAINTY_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvidenceDSTResult:
    """Output of EvidenceDSTAdapter.fuse().

    Attributes
    ----------
    combined_frame:
        The DSTFrame produced by combining all per-bundle frames via
        DSTFusion.combine_many().  This is the single belief state for
        the entire sentence's evidence pass.
    net_confidence:
        Pignistic scalar: m_true + m_unknown / 2.  Backward-compat
        bridge for callers that still expect a float confidence.
    is_genuinely_uncertain:
        True when combined_frame.m_unknown >= _UNCERTAINTY_THRESHOLD.
        Signals that the system should defer to human review.
    per_bundle_frames:
        Dict mapping predicate_id → DSTFrame, one entry per ScoredBundle
        that was successfully converted.  Used for per-claim audit.
    modal_ceiling_applied:
        Dict mapping predicate_id → ceiling float for every bundle where
        a modal_confidence_ceiling was enforced.  Empty dict if no MODAL
        predicates were present.  This is the explicit annotation required
        by spec §7 — the ceiling is surfaced here, NOT hidden inside the
        score.
    conflict_notes:
        Human-readable descriptions of any ConflictError that fired during
        combine_many().  Empty if fusion completed without conflict.
    contradiction_trace:
        Aggregated contradiction_trace strings from all ScoredBundles.
        Preserved here so stabilization / replay systems can read them
        without re-querying the individual bundles.
    stabilization_notes:
        Aggregated stabilization_notes strings from all ScoredBundles.
    """
    combined_frame: DSTFrame
    net_confidence: float
    is_genuinely_uncertain: bool
    per_bundle_frames: Dict[str, DSTFrame] = field(default_factory=dict)
    modal_ceiling_applied: Dict[str, float] = field(default_factory=dict)
    conflict_notes: List[str] = field(default_factory=list)
    contradiction_trace: List[str] = field(default_factory=list)
    stabilization_notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Lightweight dict for SSE event emission and all_sentence_data."""
        ceiling_info = (
            [
                f"{pid}: ceiling={ceil:.2f} (modal — claim asserted possibility only)"
                for pid, ceil in self.modal_ceiling_applied.items()
            ]
            if self.modal_ceiling_applied
            else []
        )
        return {
            "net_confidence": round(self.net_confidence, 6),
            "m_true": round(self.combined_frame.m_true, 6),
            "m_false": round(self.combined_frame.m_false, 6),
            "m_unknown": round(self.combined_frame.m_unknown, 6),
            "is_genuinely_uncertain": self.is_genuinely_uncertain,
            "predicate_count": len(self.per_bundle_frames),
            "modal_ceilings": ceiling_info,
            "conflicts": len(self.conflict_notes),
            "contradiction_trace_entries": len(self.contradiction_trace),
            "stabilization_note_entries": len(self.stabilization_notes),
        }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class EvidenceDSTAdapter:
    """Converts ScoredEvidence (output of EvidenceScorer) into a DSTFrame.

    Usage
    -----
    adapter = EvidenceDSTAdapter()
    result  = adapter.fuse(scored_evidence)   # EvidenceDSTResult

    The adapter is stateless and can be reused across sentences.
    """

    def __init__(self) -> None:
        self._fusion = DSTFusion()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(self, scored_evidence: Any) -> EvidenceDSTResult:
        """Convert a ScoredEvidence object into an EvidenceDSTResult.

        Parameters
        ----------
        scored_evidence:
            The object returned by EvidenceScorer.score().  Expected to
            have a .bundles attribute (list of ScoredBundle-like objects)
            and optionally a .weighted_confidence float.

        Returns
        -------
        EvidenceDSTResult
        """
        scored_bundles: List[Any] = list(
            getattr(scored_evidence, "bundles", []) or []
        )

        if not scored_bundles:
            # Nothing to fuse: return vacuous frame (total ignorance).
            vacuous = DSTFrame.vacuous(label="no_evidence")
            return EvidenceDSTResult(
                combined_frame=vacuous,
                net_confidence=DSTFusion.to_net_confidence(vacuous),
                is_genuinely_uncertain=True,
            )

        per_bundle_frames: Dict[str, DSTFrame] = {}
        modal_ceiling_applied: Dict[str, float] = {}
        contradiction_trace: List[str] = []
        stabilization_notes: List[str] = []
        conflict_notes: List[str] = []

        dst_frames: List[DSTFrame] = []

        for bundle in scored_bundles:
            predicate_id: str = self._get_predicate_id(bundle)
            frame = self._bundle_to_dst_frame(bundle, predicate_id)

            # --- MODAL ceiling enforcement (spec §7) -------------------
            ceiling: Optional[float] = getattr(
                bundle, "modal_confidence_ceiling", None
            )
            if ceiling is None:
                # Also check via the nested EvidenceBundle.frame
                raw_bundle = getattr(bundle, "bundle", None) or getattr(bundle, "evidence_bundle", None)
                if raw_bundle is not None:
                    pred_frame = getattr(raw_bundle, "frame", None)
                    if pred_frame is not None:
                        ceiling = getattr(pred_frame, "modal_certainty", None)

            if ceiling is not None:
                ceiling = float(ceiling)
                if frame.m_true > ceiling:
                    excess = frame.m_true - ceiling
                    # Absorb excess into m_unknown, not m_false —
                    # spec §7: "NOT treat the ceiling as a low-confidence signal"
                    new_m_unknown = min(1.0, frame.m_unknown + excess)
                    new_m_true = ceiling
                    new_m_false = max(0.0, round(1.0 - new_m_true - new_m_unknown, 8))
                    # Re-normalise for float drift
                    total = new_m_true + new_m_false + new_m_unknown
                    frame = DSTFrame(
                        m_true=round(new_m_true / total, 8),
                        m_false=round(new_m_false / total, 8),
                        m_unknown=round(new_m_unknown / total, 8),
                        source_label=frame.source_label,
                    )
                    modal_ceiling_applied[predicate_id] = ceiling
                    logger.debug(
                        "MODAL ceiling %.2f applied to predicate %s — "
                        "m_true capped, excess absorbed into m_unknown",
                        ceiling, predicate_id,
                    )

            per_bundle_frames[predicate_id] = frame
            dst_frames.append(frame)

            # --- Provenance aggregation --------------------------------
            for entry in list(getattr(bundle, "contradiction_trace", []) or []):
                contradiction_trace.append(str(entry))
            for note in list(getattr(bundle, "stabilization_notes", []) or []):
                stabilization_notes.append(str(note))

        # --- Combine all frames ----------------------------------------
        combined_frame = self._safe_combine_many(
            dst_frames, conflict_notes
        )

        net_confidence = DSTFusion.to_net_confidence(combined_frame)
        is_uncertain = combined_frame.m_unknown >= _UNCERTAINTY_THRESHOLD

        return EvidenceDSTResult(
            combined_frame=combined_frame,
            net_confidence=net_confidence,
            is_genuinely_uncertain=is_uncertain,
            per_bundle_frames=per_bundle_frames,
            modal_ceiling_applied=modal_ceiling_applied,
            conflict_notes=conflict_notes,
            contradiction_trace=contradiction_trace,
            stabilization_notes=stabilization_notes,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_predicate_id(bundle: Any) -> str:
        """Extract predicate_id from a ScoredBundle or EvidenceBundle."""
        # ScoredBundle exposes .predicate_id directly
        pid = getattr(bundle, "predicate_id", None)
        if pid:
            return str(pid)
        # Fall back to nested bundle.frame.predicate_id
        raw = getattr(bundle, "bundle", None) or getattr(bundle, "evidence_bundle", None)
        if raw is not None:
            frame = getattr(raw, "frame", None)
            if frame is not None:
                return str(getattr(frame, "predicate_id", ""))
        return "unknown"

    @staticmethod
    def _bundle_to_dst_frame(bundle: Any, predicate_id: str) -> DSTFrame:
        """Convert one ScoredBundle into a DSTFrame.

        Mapping (spec §7):
            m_true    = confirmation_score  (clipped [0, 1])
            m_false   = refutation_score    (clipped, must not push total > 1)
            m_unknown = 1 - m_true - m_false

        refutation_score is NOT capped by modal_certainty per Rule N6:
        "strong refutation is still strong".
        """
        raw_conf: float = float(
            getattr(bundle, "confirmation_score",
                    getattr(bundle, "net_confidence",
                            getattr(bundle, "score", 0.0))) or 0.0
        )
        raw_refut: float = float(
            getattr(bundle, "refutation_score", 0.0) or 0.0
        )

        m_true = min(1.0, max(0.0, raw_conf))
        m_false = min(1.0 - m_true, max(0.0, raw_refut))  # cannot push total > 1
        m_unknown = round(max(0.0, 1.0 - m_true - m_false), 8)

        # Guard against float drift
        total = m_true + m_false + m_unknown
        if abs(total - 1.0) > 1e-6:
            m_true    = round(m_true    / total, 8)
            m_false   = round(m_false   / total, 8)
            m_unknown = round(m_unknown / total, 8)

        return DSTFrame(
            m_true=round(m_true, 8),
            m_false=round(m_false, 8),
            m_unknown=round(m_unknown, 8),
            source_label=predicate_id,
        )

    def _safe_combine_many(
        self,
        frames: List[DSTFrame],
        conflict_notes: List[str],
    ) -> DSTFrame:
        """Combine frames, catching ConflictError gracefully.

        On ConflictError the two conflicting frames are noted in
        conflict_notes and a high-uncertainty fallback frame is returned
        so the pipeline can continue.  The fallback records m_false from
        the conflict mass so downstream can inspect it.
        """
        if not frames:
            return DSTFrame.vacuous(label="no_frames")

        result = frames[0]
        for next_frame in frames[1:]:
            try:
                result = self._fusion.combine(result, next_frame)
            except ConflictError as exc:
                note = (
                    f"ConflictError K={exc.k:.4f} between "
                    f"{exc.frame_a.source_label!r} and "
                    f"{exc.frame_b.source_label!r} — "
                    "high-uncertainty fallback applied"
                )
                conflict_notes.append(note)
                logger.warning("EvidenceDSTAdapter: %s", note)
                # Fallback: combine the two uncertainty envelopes only
                # (treat both as mostly unknown, preserve false signal)
                avg_false = (exc.frame_a.m_false + exc.frame_b.m_false) / 2.0
                remaining = max(0.0, 1.0 - avg_false)
                fallback = DSTFrame(
                    m_true=0.0,
                    m_false=round(avg_false, 8),
                    m_unknown=round(remaining, 8),
                    source_label=f"conflict_fallback({exc.frame_a.source_label},{exc.frame_b.source_label})",
                )
                result = fallback
        return result


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_adapter_instance: Optional[EvidenceDSTAdapter] = None


def get_evidence_dst_adapter() -> EvidenceDSTAdapter:
    """Return the process-level singleton EvidenceDSTAdapter.

    The adapter is stateless; the singleton just avoids re-constructing
    the DSTFusion engine on every sentence.
    """
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = EvidenceDSTAdapter()
    return _adapter_instance


# ---------------------------------------------------------------------------
# Backward-compat bridge
# ---------------------------------------------------------------------------

def adapt_expert_outputs(
    expert_outputs: Dict[str, Any],
    dst_result: EvidenceDSTResult,
) -> Dict[str, Any]:
    """Inject evidence_dst into an existing expert_outputs dict.

    Spec §7: "no existing key is overwritten".  The injection is additive.

    Parameters
    ----------
    expert_outputs:
        The dict produced by UnifiedExpertSystem / combine_routing_and_expert_decisions.
    dst_result:
        The EvidenceDSTResult from EvidenceDSTAdapter.fuse().

    Returns
    -------
    dict
        expert_outputs with 'evidence_dst' key added (or left unchanged if
        it already exists — idempotent).
    """
    if "evidence_dst" not in expert_outputs:
        expert_outputs["evidence_dst"] = dst_result.summary()
    return expert_outputs
