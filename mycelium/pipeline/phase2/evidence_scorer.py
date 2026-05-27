# mycelium/pipeline/phase2/evidence_scorer.py
# Stage 8 — Re-ranking and epistemic burden weighting for evidence bundles.
#
# EvidenceScorer accepts an EvidenceResult (from Stage 7 EvidenceFinder)
# and produces a ScoredEvidenceResult by computing a composite_score for
# every EvidenceItem and a bundle_confidence for every EvidenceBundle.
#
# No I/O, no retrieval, no LLM calls.  Pure arithmetic over the data
# types defined in evidence_types.py and predicate_types.py.
#
# Spec ref: implementation spec v0.2.1 — Section 8.2 (Evidence Scoring)

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mycelium.pipeline.phase2.evidence_types import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceResult,
    RetrievalMethod,
)
from mycelium.pipeline.predicates.predicate_types import (
    EpistemicBurden,
    ModalCertainty,
    PredicateFrame,
    PredicateType,
)


# ---------------------------------------------------------------------------
# Score weights (injectable for domain tuning)
# ---------------------------------------------------------------------------

@dataclass
class ScoreWeights:
    """Multipliers applied during composite score calculation.

    All burden multipliers must be in [0.0, 2.0].  Defaults match the
    spec v0.2.1 Section 8.2 table.

    Attributes
    ----------
    burden_none:          Baseline for FACTIVE / unweighted frames.
    burden_scope_bounded: Evidence must match the scope to be valid.
    burden_exhaustive:    Strong burden but still falsifiable.
    burden_rebuttal:      Counter-evidence boost (high epistemic value).
    burden_modal:         Possibility claims are softer evidence.
    certainty_certain:    MODAL frame with CERTAIN certainty.
    certainty_probable:   MODAL frame with PROBABLE certainty.
    certainty_possible:   MODAL frame with POSSIBLE certainty.
    certainty_speculative: MODAL frame with SPECULATIVE certainty.
    certainty_unknown:    MODAL frame with unknown certainty.
    """
    burden_none: float = 1.00
    burden_scope_bounded: float = 0.85
    burden_exhaustive: float = 0.90
    burden_rebuttal: float = 1.10
    burden_modal: float = 0.70

    certainty_certain: float = 1.00
    certainty_probable: float = 0.90
    certainty_possible: float = 0.75
    certainty_speculative: float = 0.60
    certainty_unknown: float = 0.70


_DEFAULT_WEIGHTS = ScoreWeights()


# ---------------------------------------------------------------------------
# Scored data types
# ---------------------------------------------------------------------------

@dataclass
class ScoredItem:
    """An EvidenceItem annotated with its composite score.

    Attributes
    ----------
    item:
        The original EvidenceItem from EvidenceFinder.
    composite_score:
        Final score after applying burden and certainty multipliers.
        Clamped to [0.0, 1.0].
    score_breakdown:
        Dict recording every multiplier applied, for auditability:
        {
          'raw_score':              float,
          'burden_multiplier':      float,
          'certainty_multiplier':   float,
          'composite_score':        float,
          'burden':                 str,
          'certainty':              str | None,
        }
    """
    item: EvidenceItem
    composite_score: float
    score_breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredBundle:
    """An EvidenceBundle annotated with scored items and bundle confidence.

    Attributes
    ----------
    bundle:
        The original EvidenceBundle from EvidenceFinder.
    scored_items:
        ScoredItems sorted descending by composite_score.
    bundle_confidence:
        Mean composite_score over all items with composite_score > 0.
        0.0 when scored_items is empty or all items score 0.
    """
    bundle: EvidenceBundle
    scored_items: List[ScoredItem]
    bundle_confidence: float

    def best(self) -> Optional[ScoredItem]:
        """Return the highest composite_score ScoredItem, or None."""
        return self.scored_items[0] if self.scored_items else None

    def above_threshold(self, threshold: float = 0.5) -> List[ScoredItem]:
        """Return ScoredItems with composite_score >= threshold."""
        return [s for s in self.scored_items if s.composite_score >= threshold]


@dataclass
class ScoredEvidenceResult:
    """Top-level result from EvidenceScorer.score().

    Produced after Stage 8 and attached to the workflow context as
    context.scored_evidence.  Phase 2.5 (CalibrationPipeline) reads
    weighted_confidence and scored_bundles.

    Attributes
    ----------
    scored_bundles:
        One ScoredBundle per falsifiable frame, preserving Stage 7 order.
    weighted_confidence:
        Pipeline-level confidence: weighted mean bundle_confidence where
        the weight of each bundle is its item count.  Bundles with no
        items (coverage gap) contribute weight 0.  In [0.0, 1.0].
    total_ms:
        Wall-clock time in ms for the entire score() call.
    store_summary:
        Carried from EvidenceResult.store_summary for SSE emission.
    skipped_ids:
        predicate_ids skipped upstream (NORMATIVE frames).
    """
    scored_bundles: List[ScoredBundle]
    weighted_confidence: float
    total_ms: float
    store_summary: Dict[str, Any]
    skipped_ids: List[str] = field(default_factory=list)

    def bundle_for(self, predicate_id: str) -> Optional[ScoredBundle]:
        """Return the ScoredBundle for the given predicate_id, or None."""
        for sb in self.scored_bundles:
            if sb.bundle.frame.predicate_id == predicate_id:
                return sb
        return None

    def all_scored_items(self) -> List[ScoredItem]:
        """Flatten all ScoredItems across all bundles."""
        result = []
        for sb in self.scored_bundles:
            result.extend(sb.scored_items)
        return result

    def summary(self) -> Dict[str, Any]:
        """Lightweight dict for SSE event emission."""
        return {
            "scored_bundles": len(self.scored_bundles),
            "total_scored_items": len(self.all_scored_items()),
            "weighted_confidence": round(self.weighted_confidence, 4),
            "total_ms": round(self.total_ms, 2),
            "skipped": len(self.skipped_ids),
            "store": self.store_summary,
        }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _burden_multiplier(frame: PredicateFrame, weights: ScoreWeights) -> tuple[float, str]:
    """Return (multiplier, label) for the frame's epistemic burden."""
    burden = getattr(frame, "epistemic_burden", EpistemicBurden.NONE)
    mapping: dict[EpistemicBurden, tuple[float, str]] = {
        EpistemicBurden.NONE: (weights.burden_none, "NONE"),
        EpistemicBurden.SCOPE_BOUNDED: (weights.burden_scope_bounded, "SCOPE_BOUNDED"),
        EpistemicBurden.EXHAUSTIVE: (weights.burden_exhaustive, "EXHAUSTIVE"),
        EpistemicBurden.REBUTTAL: (weights.burden_rebuttal, "REBUTTAL"),
        EpistemicBurden.MODAL: (weights.burden_modal, "MODAL"),
    }
    return mapping.get(burden, (weights.burden_none, "NONE"))


def _certainty_multiplier(
    frame: PredicateFrame,
    weights: ScoreWeights,
) -> tuple[float, Optional[str]]:
    """Return (multiplier, label) for MODAL frames; (1.0, None) otherwise."""
    if getattr(frame, "predicate_type", None) != PredicateType.MODAL:
        return 1.0, None
    certainty = getattr(frame, "modal_certainty", None)
    if certainty is None:
        return weights.certainty_unknown, "UNKNOWN"
    mapping: dict[ModalCertainty, tuple[float, str]] = {
        ModalCertainty.CERTAIN: (weights.certainty_certain, "CERTAIN"),
        ModalCertainty.PROBABLE: (weights.certainty_probable, "PROBABLE"),
        ModalCertainty.POSSIBLE: (weights.certainty_possible, "POSSIBLE"),
        ModalCertainty.SPECULATIVE: (weights.certainty_speculative, "SPECULATIVE"),
    }
    return mapping.get(certainty, (weights.certainty_unknown, "UNKNOWN"))


def _score_item(
    item: EvidenceItem,
    frame: PredicateFrame,
    weights: ScoreWeights,
) -> ScoredItem:
    """Compute composite_score for a single EvidenceItem."""
    raw = item.relevance_score
    b_mult, b_label = _burden_multiplier(frame, weights)
    c_mult, c_label = _certainty_multiplier(frame, weights)
    composite = raw * b_mult * c_mult
    # Clamp to [0.0, 1.0]
    composite = max(0.0, min(1.0, composite))
    breakdown: Dict[str, Any] = {
        "raw_score": raw,
        "burden_multiplier": b_mult,
        "certainty_multiplier": c_mult,
        "composite_score": composite,
        "burden": b_label,
        "certainty": c_label,
    }
    return ScoredItem(item=item, composite_score=composite, score_breakdown=breakdown)


def _bundle_confidence(scored_items: List[ScoredItem]) -> float:
    """Mean composite_score over items with composite_score > 0."""
    positive = [s.composite_score for s in scored_items if s.composite_score > 0.0]
    if not positive:
        return 0.0
    return sum(positive) / len(positive)


def _weighted_confidence(scored_bundles: List[ScoredBundle]) -> float:
    """Weighted mean bundle_confidence; weight = item count.

    Bundles with no items contribute weight 0 (coverage gap).
    Returns 0.0 when total weight is 0.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for sb in scored_bundles:
        w = float(len(sb.scored_items))
        total_weight += w
        weighted_sum += sb.bundle_confidence * w
    if total_weight == 0.0:
        return 0.0
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# EvidenceScorer
# ---------------------------------------------------------------------------

class EvidenceScorer:
    """Re-rank and burden-weight evidence bundles from EvidenceFinder.

    Typical usage
    -------------
    ::
        from mycelium.pipeline.phase2.evidence_scorer import get_evidence_scorer

        scored = get_evidence_scorer().score(context.evidence_result)
        context.scored_evidence = scored
        # scored.weighted_confidence -> float passed to CalibrationPipeline

    Constructor
    -----------
    Accepts optional ScoreWeights for per-domain tuning::

        scorer = EvidenceScorer(weights=ScoreWeights(burden_modal=0.50))
    """

    def __init__(self, weights: Optional[ScoreWeights] = None) -> None:
        self._weights = weights or _DEFAULT_WEIGHTS

    # ------------------------------------------------------------------
    # Per-bundle scoring
    # ------------------------------------------------------------------

    def score_bundle(self, bundle: EvidenceBundle) -> ScoredBundle:
        """Score all items in a single EvidenceBundle.

        Parameters
        ----------
        bundle:
            An EvidenceBundle from EvidenceFinder.find().

        Returns
        -------
        ScoredBundle with scored_items sorted descending by composite_score
        and bundle_confidence set.
        """
        frame = bundle.frame
        scored_items = [
            _score_item(item, frame, self._weights)
            for item in bundle.items
        ]
        scored_items.sort(key=lambda s: s.composite_score, reverse=True)
        confidence = _bundle_confidence(scored_items)
        return ScoredBundle(
            bundle=bundle,
            scored_items=scored_items,
            bundle_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Full-result scoring
    # ------------------------------------------------------------------

    def score(
        self,
        evidence_result: EvidenceResult,
    ) -> ScoredEvidenceResult:
        """Score every bundle in an EvidenceResult.

        NORMATIVE frames were already excluded by EvidenceFinder and
        are carried through skipped_ids without re-processing.

        Parameters
        ----------
        evidence_result:
            EvidenceResult from EvidenceFinder.find().

        Returns
        -------
        ScoredEvidenceResult ready for Phase 2.5 CalibrationPipeline.
        Never raises.
        """
        t0 = time.perf_counter()

        scored_bundles = [
            self.score_bundle(bundle)
            for bundle in evidence_result.bundles
        ]
        w_confidence = _weighted_confidence(scored_bundles)

        total_ms = (time.perf_counter() - t0) * 1000.0
        return ScoredEvidenceResult(
            scored_bundles=scored_bundles,
            weighted_confidence=w_confidence,
            total_ms=total_ms,
            store_summary=evidence_result.store_summary,
            skipped_ids=list(evidence_result.skipped_ids),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        w = self._weights
        return (
            f"EvidenceScorer(burden_none={w.burden_none}, "
            f"burden_modal={w.burden_modal}, "
            f"burden_rebuttal={w.burden_rebuttal})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_scorer_instance: Optional[EvidenceScorer] = None


def get_evidence_scorer() -> EvidenceScorer:
    """Return the module-level EvidenceScorer singleton.

    Lazily instantiated on first call with default ScoreWeights.
    """
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = EvidenceScorer()
    return _scorer_instance
