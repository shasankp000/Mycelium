"""
Layer 5 — Hypothesis Evaluator
================================
Takes the output of Layer 3 (PredicateGenerator) and Layer 4
(EvidenceGrounder) and produces a verdict about the query's core
hypothesis.

The hypothesis is treated as the query's positive_predicates set:
"Are the positive claims in this query supported by available evidence?"

Verdict schema
--------------
    SUPPORTED        — ≥2/3 of positive predicates are grounded with
                        evidence_score >= confidence_threshold
    REFUTED          — ≥2/3 of negative predicates conflict with
                        grounded positive predicates on the same subject
    INCONCLUSIVE     — insufficient evidence to decide either way
    CONTRADICTORY    — both SUPPORTED and REFUTED signals fire equally;
                        the query itself is internally inconsistent
    UNCERTAIN        — fallback (fewer than min_predicates to evaluate)

DST integration
---------------
    When mycelium.fusion.dst_fusion.DSTFusion is available, the evaluator
    computes a proper DST belief frame instead of the scalar heuristics.
    The m_true mass maps to SUPPORTED, m_false to REFUTED, m_unknown to
    INCONCLUSIVE, and m_conflict to CONTRADICTORY.
    This gives the verdict Dempster-Shafer epistemic grounding, consistent
    with Phase F of the Mycelium philosophy (uncertain evidence ≠ false
    evidence).

When DST is unavailable the evaluator falls back to the scalar heuristic
(proportion of grounded predicates above threshold).

Output dict
-----------
    verdict          : str    — one of the five verdict labels above
    confidence       : float  — [0,1] confidence in the verdict
    support_score    : float  — proportion of positive preds grounded
    refutation_score : float  — proportion of negative preds that
                                  oppose grounded positive preds
    n_evaluated      : int
    reasoning        : str    — human-readable explanation
    dst_frame        : dict   — {m_true, m_false, m_unknown, m_conflict}
                                 if DST was used; None otherwise
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# DST is optional — graceful degrade
try:
    from mycelium.fusion.dst_fusion import DSTFusion
    _DST_AVAILABLE = True
except ImportError:
    DSTFusion = None  # type: ignore
    _DST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------
VERDICT_SUPPORTED     = "SUPPORTED"
VERDICT_REFUTED       = "REFUTED"
VERDICT_INCONCLUSIVE  = "INCONCLUSIVE"
VERDICT_CONTRADICTORY = "CONTRADICTORY"
VERDICT_UNCERTAIN     = "UNCERTAIN"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HypothesisEvaluator:
    """
    Layer 5: evaluate the core hypothesis of an OOD query.

    Parameters
    ----------
    confidence_threshold : float
        Minimum evidence_score for a predicate to count as grounded
        when using the scalar fallback (default 0.40).
    min_predicates : int
        Minimum number of predicates needed to issue a verdict other
        than UNCERTAIN (default 2).
    use_dst : bool
        Whether to attempt DST integration when DSTFusion is available
        (default True).  Set to False for purely scalar evaluation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        min_predicates: int = 2,
        use_dst: bool = True,
    ) -> None:
        self._thresh = confidence_threshold
        self._min_preds = min_predicates
        self._use_dst = use_dst and _DST_AVAILABLE
        self._dst = DSTFusion() if self._use_dst else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predicates: Dict,
        evidence: Dict,
        dag_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Evaluate the hypothesis encoded in the predicate sets.

        Parameters
        ----------
        predicates : dict
            Output of PredicateGenerator.generate().
        evidence : dict
            Output of EvidenceGrounder.ground().
        dag_context : dict, optional
            Original dag_context (used to pick up contradiction_signal).

        Returns
        -------
        dict  — see module docstring for schema.
        """
        grounded_preds: List[Dict] = evidence.get("grounded_predicates", [])
        positive_preds = [p for p in grounded_preds if p.get("polarity") == "POSITIVE"]
        negative_preds = [p for p in grounded_preds if p.get("polarity") == "NEGATIVE"]
        all_preds      = grounded_preds

        if len(all_preds) < self._min_preds:
            return self._uncertain_result(len(all_preds))

        # Check contradiction_signal from Layer 2
        contradiction_signal = (
            dag_context.get("contradiction_signal")
            if dag_context else None
        )

        if self._use_dst and self._dst is not None:
            return self._evaluate_dst(
                positive_preds, negative_preds, contradiction_signal
            )
        return self._evaluate_scalar(
            positive_preds, negative_preds, contradiction_signal
        )

    # ------------------------------------------------------------------
    # DST path
    # ------------------------------------------------------------------

    def _evaluate_dst(
        self,
        positive_preds: List[Dict],
        negative_preds: List[Dict],
        contradiction_signal: Optional[Dict],
    ) -> Dict:
        """
        Build a DST belief frame from the evidence scores.

        Each grounded positive predicate contributes m_true mass.
        Each grounded negative predicate contributes m_false mass.
        Ungrounded predicates contribute m_unknown.
        The contradiction_signal (from Phase E) contributes m_conflict.
        """
        try:
            import math

            def _mean_score(preds: List[Dict]) -> float:
                scores = [p.get("evidence_score", 0.0) for p in preds]
                return sum(scores) / len(scores) if scores else 0.0

            pos_score = _mean_score([p for p in positive_preds if p.get("grounded")])
            neg_score = _mean_score([p for p in negative_preds if p.get("grounded")])

            n_ungrounded = sum(
                1 for p in positive_preds + negative_preds
                if not p.get("grounded")
            )
            n_total = len(positive_preds) + len(negative_preds)
            unknown_frac = n_ungrounded / n_total if n_total > 0 else 1.0

            conflict_mass = 0.0
            if contradiction_signal is not None:
                conflict_mass = min(0.3, contradiction_signal.get("severity", 0.0))

            # Normalise to [0,1] summing to 1
            raw_true    = pos_score * (1.0 - unknown_frac) * (1.0 - conflict_mass)
            raw_false   = neg_score * (1.0 - unknown_frac) * (1.0 - conflict_mass)
            raw_unknown = unknown_frac * (1.0 - conflict_mass)
            total = raw_true + raw_false + raw_unknown + conflict_mass
            if total < 1e-9:
                total = 1.0

            frame = {
                "m_true":    raw_true    / total,
                "m_false":   raw_false   / total,
                "m_unknown": raw_unknown / total,
                "m_conflict": conflict_mass / total,
            }

            verdict, confidence = self._verdict_from_frame(frame)
            reasoning = (
                f"DST frame: m_true={frame['m_true']:.3f} "
                f"m_false={frame['m_false']:.3f} "
                f"m_unknown={frame['m_unknown']:.3f} "
                f"m_conflict={frame['m_conflict']:.3f}. "
                f"Verdict: {verdict}."
            )

            return {
                "verdict":          verdict,
                "confidence":       confidence,
                "support_score":    frame["m_true"],
                "refutation_score": frame["m_false"],
                "n_evaluated":      len(positive_preds) + len(negative_preds),
                "reasoning":        reasoning,
                "dst_frame":        frame,
            }

        except Exception as exc:
            logger.warning("HypothesisEvaluator DST path failed (%s), falling back", exc)
            return self._evaluate_scalar(
                positive_preds, negative_preds, contradiction_signal
            )

    # ------------------------------------------------------------------
    # Scalar heuristic path
    # ------------------------------------------------------------------

    def _evaluate_scalar(
        self,
        positive_preds: List[Dict],
        negative_preds: List[Dict],
        contradiction_signal: Optional[Dict],
    ) -> Dict:
        n_pos = len(positive_preds)
        n_neg = len(negative_preds)

        # Support score: fraction of positive preds grounded above threshold
        grounded_pos = [
            p for p in positive_preds
            if p.get("grounded") and p.get("evidence_score", 0.0) >= self._thresh
        ]
        support_score = len(grounded_pos) / n_pos if n_pos > 0 else 0.0

        # Refutation score: fraction of negative preds grounded above threshold
        grounded_neg = [
            p for p in negative_preds
            if p.get("grounded") and p.get("evidence_score", 0.0) >= self._thresh
        ]
        refutation_score = len(grounded_neg) / n_neg if n_neg > 0 else 0.0

        # Contradiction adjustment
        if contradiction_signal is not None:
            conflict_boost = contradiction_signal.get("severity", 0.0) * 0.3
            support_score    = max(0.0, support_score    - conflict_boost)
            refutation_score = max(0.0, refutation_score - conflict_boost)

        both_high = support_score >= 0.5 and refutation_score >= 0.5
        if both_high:
            verdict    = VERDICT_CONTRADICTORY
            confidence = 0.5
        elif support_score >= 0.66:
            verdict    = VERDICT_SUPPORTED
            confidence = support_score
        elif refutation_score >= 0.66:
            verdict    = VERDICT_REFUTED
            confidence = refutation_score
        elif support_score > 0.0 or refutation_score > 0.0:
            verdict    = VERDICT_INCONCLUSIVE
            confidence = max(support_score, refutation_score)
        else:
            verdict    = VERDICT_UNCERTAIN
            confidence = 0.0

        reasoning = (
            f"Scalar: support={support_score:.2f}, refutation={refutation_score:.2f}, "
            f"pos={n_pos}, neg={n_neg}. Verdict: {verdict}."
        )

        logger.debug("HypothesisEvaluator: %s", reasoning)

        return {
            "verdict":          verdict,
            "confidence":       confidence,
            "support_score":    support_score,
            "refutation_score": refutation_score,
            "n_evaluated":      n_pos + n_neg,
            "reasoning":        reasoning,
            "dst_frame":        None,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _uncertain_result(self, n: int) -> Dict:
        return {
            "verdict":          VERDICT_UNCERTAIN,
            "confidence":       0.0,
            "support_score":    0.0,
            "refutation_score": 0.0,
            "n_evaluated":      n,
            "reasoning":        f"Only {n} predicates found; need >= {self._min_preds}.",
            "dst_frame":        None,
        }

    @staticmethod
    def _verdict_from_frame(frame: Dict) -> tuple[str, float]:
        """Map a DST frame to a (verdict, confidence) pair."""
        if frame["m_conflict"] >= 0.25:
            return VERDICT_CONTRADICTORY, frame["m_conflict"]
        if frame["m_true"] >= 0.50:
            return VERDICT_SUPPORTED, frame["m_true"]
        if frame["m_false"] >= 0.50:
            return VERDICT_REFUTED, frame["m_false"]
        if frame["m_unknown"] >= 0.60:
            return VERDICT_INCONCLUSIVE, frame["m_unknown"]
        # Weak signals — pick dominant
        dominant = max(frame, key=frame.get)
        label_map = {
            "m_true":    VERDICT_SUPPORTED,
            "m_false":   VERDICT_REFUTED,
            "m_unknown": VERDICT_INCONCLUSIVE,
            "m_conflict": VERDICT_CONTRADICTORY,
        }
        return label_map[dominant], frame[dominant]
