"""
Phase E — Contradiction Classifier
====================================
Seven-class semantic contradiction classifier.

Consolidation Notes §18, §32.

The §18 Invariant (THE MOST CRITICAL CONSTRAINT IN THIS FILE):
---------------------------------------------------------------
    "Coffee improves focus" vs "Coffee increases anxiety"
    MUST classify as TRADEOFF_RELATION, NEVER DIRECT_CONTRADICTION.

    These two claims are simultaneously satisfiable:
        - One describes a cognitive benefit.
        - One describes a physiological cost.
        - They are true for the same substance in the same human.
        - No logical law prevents both being true at once.

    DIRECT_CONTRADICTION requires mutual exclusivity.
    TRADEOFF_RELATION is for dual-aspect claims on the same subject.

The seven classes (from §32, implemented in strict pipeline order):
    TEMPORAL_CONTRADICTION     — contradictory only across time
    DIRECT_CONTRADICTION       — same predicate + scope, mutually exclusive
    PARTIAL_CONTRADICTION      — same predicate, different scope/degree
    CONTEXTUAL_CONTRADICTION   — contradictory in a specific context only
    PROBABILISTIC_DISAGREEMENT — statistical/probabilistic tension
    TRADEOFF_RELATION          — both true but in tension (§18 coffee case)
    NON_CONTRADICTORY_DIVERGENCE — merely different, not contradictory

Classification pipeline (7 stages):
    1. Temporal gate          ← non-overlapping temporal windows
    2. Polarity inversion gate ← negation of identical predicate+subject
    3. Scope narrowing gate   ← same predicate, different scope
    4. Embedding similarity gate ← too dissimilar to be contradictory
    5. Tradeoff gate          ← simultaneous-satisfiability test (§18)
    6. Context gate           ← context qualifiers differ
    7. Probabilistic gate     ← low-confidence claims
    Fallback                  ← NON_CONTRADICTORY_DIVERGENCE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# §18 Invariant: aspect keywords that indicate dual-aspect (tradeoff) claims.
# A claim pair is TRADEOFF if BOTH subjects are the same AND their objects
# come from DIFFERENT aspect categories.  Adding new substances or aspects
# here extends the tradeoff detection without touching the pipeline logic.
# ---------------------------------------------------------------------------
_TRADEOFF_ASPECT_GROUPS: List[List[str]] = [
    # cognitive / physiological aspects of stimulants (§18 coffee example)
    ["focus", "concentration", "alertness", "productivity", "cognition",
     "memory", "attention", "performance"],
    ["anxiety", "stress", "jitters", "nervousness", "heart rate",
     "blood pressure", "palpitation", "insomnia", "restlessness"],
    # benefit / cost framing (generic)
    ["benefit", "advantage", "improvement", "gain", "enhancement",
     "positive effect", "helps", "boosts", "increases", "improves"],
    ["risk", "harm", "side effect", "drawback", "cost", "negative effect",
     "worsens", "reduces", "impairs", "damages"],
    # causal cost/benefit split in medical claims
    ["treats", "cures", "prevents", "reduces risk of", "lowers"],
    ["causes", "triggers", "induces", "worsens", "increases risk of",
     "elevates"],
]

# Classification thresholds — all in one place, no magic numbers in logic.
THRESHOLDS: Dict[str, float] = {
    "direct_min_sim": 0.70,      # cosine sim floor to be contradictory at all
    "tradeoff_sim": 0.50,        # lower floor because tradeoffs share a subject
    "prob_confidence_cap": 0.60, # below this → probabilistic_disagreement
    "high_severity": 0.80,       # severity returned for DIRECT_CONTRADICTION
    "medium_severity": 0.55,     # severity for PARTIAL / CONTEXTUAL
    "low_severity": 0.30,        # severity for TRADEOFF / PROBABILISTIC
    "divergence_severity": 0.10, # severity for NON_CONTRADICTORY_DIVERGENCE
}

# Scope vocabulary: subject/object tokens that imply limited scope.
_SCOPE_LIMITERS = [
    "some", "certain", "in some cases", "sometimes", "often", "typically",
    "under conditions", "in context", "for most", "for some",
    "in the short term", "in the long term", "temporarily", "chronically",
]

# Negation tokens for polarity inversion detection (Stage 2).
_NEGATIONS = [
    "not", "no", "never", "neither", "nor", "cannot", "can't", "won't",
    "doesn't", "does not", "is not", "isn't", "are not", "aren't",
    "fails to", "unable to", "lack", "lacks", "without",
]


@dataclass
class ClassificationResult:
    """Output of ContradictionClassifier.classify().

    Attributes
    ----------
    contradiction_type : str
        One of the seven classes from CONTRADICTION_CLASSES.
    severity : float
        Contradiction strength [0, 1].  Passed to TRMEngine.record_contradiction.
    confidence : float
        Classifier confidence in the assigned type [0, 1].
    scope : str
        LOCAL | DOMAIN | GLOBAL.
    explanation : str
        Human-readable rationale for the classification.
    stage_reached : int
        Which pipeline stage (1-7) produced the classification.
        7+ means the fallback was used.
    evidence : dict
        Stage-specific evidence (similarity scores, matched tokens, etc.).
    """

    contradiction_type: str
    severity: float
    confidence: float
    scope: str
    explanation: str
    stage_reached: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)


class ContradictionClassifier:
    """Seven-class semantic contradiction classifier.

    Parameters
    ----------
    embedding_fn : callable, optional
        (str) -> list[float].  Required for Stage 4 embedding gate.
        Without it, Stage 4 is skipped and the pipeline continues.

    Usage
    -----
    clf = ContradictionClassifier(embedding_fn=my_embed)
    result = clf.classify(node_a, node_b)
    # result.contradiction_type in CONTRADICTION_CLASSES
    """

    def __init__(
        self, *, embedding_fn: Optional[Any] = None
    ) -> None:
        self._embed = embedding_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self, node_a: Any, node_b: Any
    ) -> ClassificationResult:
        """Classify the contradiction relationship between two IRNodes.

        The pipeline is strictly ordered and short-circuits at the first
        confident match.  NON_CONTRADICTORY_DIVERGENCE is the safe fallback
        (§32: never over-classify).

        Parameters
        ----------
        node_a, node_b : IRNode
            The two claim nodes to compare.

        Returns
        -------
        ClassificationResult
        """
        label_a = node_a.label.lower()
        label_b = node_b.label.lower()
        sig_a = node_a.semantic_signature
        sig_b = node_b.semantic_signature

        # Stage 1: temporal gate
        result = self._stage_temporal(node_a, node_b)
        if result:
            return result

        # Stage 2: polarity inversion gate
        result = self._stage_polarity(label_a, label_b, sig_a, sig_b)
        if result:
            return result

        # Stage 3: scope narrowing gate
        result = self._stage_scope(label_a, label_b, sig_a, sig_b)
        if result:
            return result

        # Stage 4: embedding similarity gate (requires embedding_fn)
        sim: Optional[float] = None
        if self._embed is not None:
            sim = self._cosine_sim(
                sig_a.embedding_signature, sig_b.embedding_signature
            )
            result = self._stage_embedding(
                label_a, label_b, sig_a, sig_b, sim
            )
            if result:
                return result

        # Stage 5: tradeoff gate (§18 invariant MUST fire before Stage 6)
        result = self._stage_tradeoff(label_a, label_b, sig_a, sig_b, sim)
        if result:
            return result

        # Stage 6: context gate
        result = self._stage_context(label_a, label_b, sig_a, sig_b)
        if result:
            return result

        # Stage 7: probabilistic gate
        result = self._stage_probabilistic(node_a, node_b)
        if result:
            return result

        # Fallback
        return ClassificationResult(
            contradiction_type="NON_CONTRADICTORY_DIVERGENCE",
            severity=THRESHOLDS["divergence_severity"],
            confidence=0.9,
            scope="LOCAL",
            explanation="No significant contradiction pattern detected.",
            stage_reached=8,
            evidence={},
        )

    def check_invariant_18(self, node_a: Any, node_b: Any) -> bool:
        """Verify that the §18 invariant holds for a given node pair.

        Returns True if the pair IS a TRADEOFF (simultaneous-satisfiable),
        meaning the classifier MUST NOT return DIRECT_CONTRADICTION for it.

        This method is public so that tests can assert the invariant
        directly without going through the full classify() pipeline.
        """
        label_a = node_a.label.lower()
        label_b = node_b.label.lower()
        sig_a = node_a.semantic_signature
        sig_b = node_b.semantic_signature
        result = self._stage_tradeoff(label_a, label_b, sig_a, sig_b, sim=None)
        return result is not None

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_temporal(
        self, node_a: Any, node_b: Any
    ) -> Optional[ClassificationResult]:
        """Stage 1: non-overlapping temporal windows → TEMPORAL_CONTRADICTION."""
        try:
            ts_a = node_a.temporal_state
            ts_b = node_b.temporal_state
        except AttributeError:
            return None

        # If historical_validity differs: one claim is no longer true.
        if (
            ts_a.type not in ("UNKNOWN",)
            and ts_b.type not in ("UNKNOWN",)
            and ts_a.historical_validity != ts_b.historical_validity
        ):
            return ClassificationResult(
                contradiction_type="TEMPORAL_CONTRADICTION",
                severity=THRESHOLDS["medium_severity"],
                confidence=0.75,
                scope="DOMAIN",
                explanation=(
                    f"Claims have different historical validity: "
                    f"node_a.historical_validity={ts_a.historical_validity}, "
                    f"node_b.historical_validity={ts_b.historical_validity}."
                ),
                stage_reached=1,
                evidence={
                    "ts_a_type": ts_a.type,
                    "ts_b_type": ts_b.type,
                    "historical_validity_a": ts_a.historical_validity,
                    "historical_validity_b": ts_b.historical_validity,
                },
            )

        # INTERVAL types with non-overlapping windows
        if (
            ts_a.type == "INTERVAL"
            and ts_b.type == "INTERVAL"
            and ts_a.end is not None
            and ts_b.start is not None
            and ts_a.end < ts_b.start
        ):
            return ClassificationResult(
                contradiction_type="TEMPORAL_CONTRADICTION",
                severity=THRESHOLDS["medium_severity"],
                confidence=0.85,
                scope="DOMAIN",
                explanation=(
                    f"Non-overlapping temporal intervals: "
                    f"A ends {ts_a.end}, B starts {ts_b.start}."
                ),
                stage_reached=1,
                evidence={"ts_a_end": ts_a.end, "ts_b_start": ts_b.start},
            )

        return None

    def _stage_polarity(
        self,
        label_a: str,
        label_b: str,
        sig_a: Any,
        sig_b: Any,
    ) -> Optional[ClassificationResult]:
        """Stage 2: polarity inversion → DIRECT_CONTRADICTION.

        Fires only when:
        - Same predicate family
        - Same subject tokens (overlap >= 0.5 Jaccard on words)
        - One label contains a negation token that the other lacks

        The §18 tradeoff invariant is NOT checked here because Stage 5
        fires after Stage 2 in the pipeline.  If Stage 2 returns a
        DIRECT_CONTRADICTION for a tradeoff pair, Stage 5 can never
        reach it.  To prevent this, we add an early simultaneous-
        satisfiability check WITHIN Stage 2.
        """
        # Same predicate family required
        if sig_a.predicate_family != sig_b.predicate_family:
            return None

        # Subject similarity: Jaccard on words
        words_a = set(label_a.split())
        words_b = set(label_b.split())
        union = words_a | words_b
        if not union:
            return None
        jaccard = len(words_a & words_b) / len(union)
        if jaccard < 0.3:
            return None  # too dissimilar to be direct contradiction

        neg_in_a = any(neg in label_a for neg in _NEGATIONS)
        neg_in_b = any(neg in label_b for neg in _NEGATIONS)

        # Polarity inversion: exactly one has a negation
        if neg_in_a == neg_in_b:
            return None

        # §18 guard inside Stage 2: if this is a tradeoff pair, do not
        # classify as DIRECT_CONTRADICTION regardless of polarity.
        if self._is_tradeoff_pair(label_a, label_b):
            logger.debug(
                "Stage 2: polarity inversion detected BUT §18 tradeoff guard fired —"
                " deferring to Stage 5."
            )
            return None

        matched_neg = (
            [n for n in _NEGATIONS if n in label_a]
            if neg_in_a
            else [n for n in _NEGATIONS if n in label_b]
        )

        return ClassificationResult(
            contradiction_type="DIRECT_CONTRADICTION",
            severity=THRESHOLDS["high_severity"],
            confidence=0.82,
            scope="DOMAIN" if jaccard > 0.6 else "LOCAL",
            explanation=(
                f"Polarity inversion detected. Same predicate family "
                f"({sig_a.predicate_family}), Jaccard={jaccard:.2f}. "
                f"Negation tokens: {matched_neg}."
            ),
            stage_reached=2,
            evidence={
                "predicate_family": sig_a.predicate_family,
                "jaccard": jaccard,
                "negations_matched": matched_neg,
            },
        )

    def _stage_scope(
        self,
        label_a: str,
        label_b: str,
        sig_a: Any,
        sig_b: Any,
    ) -> Optional[ClassificationResult]:
        """Stage 3: same predicate, different scope → PARTIAL_CONTRADICTION."""
        if sig_a.predicate_family != sig_b.predicate_family:
            return None

        # Scope limiter present in one but not both
        scoped_a = any(s in label_a for s in _SCOPE_LIMITERS)
        scoped_b = any(s in label_b for s in _SCOPE_LIMITERS)

        if scoped_a != scoped_b:   # exactly one is scoped
            matched_limiters = (
                [s for s in _SCOPE_LIMITERS if s in label_a]
                if scoped_a
                else [s for s in _SCOPE_LIMITERS if s in label_b]
            )
            return ClassificationResult(
                contradiction_type="PARTIAL_CONTRADICTION",
                severity=THRESHOLDS["medium_severity"],
                confidence=0.70,
                scope="LOCAL",
                explanation=(
                    f"Same predicate family ({sig_a.predicate_family}) "
                    f"but scope differs. Limiters: {matched_limiters}."
                ),
                stage_reached=3,
                evidence={
                    "predicate_family": sig_a.predicate_family,
                    "scope_limiters_matched": matched_limiters,
                },
            )

        return None

    def _stage_embedding(
        self,
        label_a: str,
        label_b: str,
        sig_a: Any,
        sig_b: Any,
        sim: float,
    ) -> Optional[ClassificationResult]:
        """Stage 4: embedding similarity gate.

        If similarity is below direct_min_sim AND above tradeoff_sim,
        the claims are too different to constitute a contradiction.
        Returns NON_CONTRADICTORY_DIVERGENCE only if similarity is truly
        low (below tradeoff_sim floor).
        """
        if sim is None:
            return None

        if sim < THRESHOLDS["tradeoff_sim"]:
            return ClassificationResult(
                contradiction_type="NON_CONTRADICTORY_DIVERGENCE",
                severity=THRESHOLDS["divergence_severity"],
                confidence=0.85,
                scope="LOCAL",
                explanation=(
                    f"Embedding similarity too low ({sim:.3f} < "
                    f"{THRESHOLDS['tradeoff_sim']}) — claims are "
                    f"unrelated, not contradictory."
                ),
                stage_reached=4,
                evidence={"cosine_similarity": sim},
            )

        return None

    def _stage_tradeoff(
        self,
        label_a: str,
        label_b: str,
        sig_a: Any,
        sig_b: Any,
        sim: Optional[float],
    ) -> Optional[ClassificationResult]:
        """Stage 5: simultaneous-satisfiability test (§18 invariant).

        This stage MUST fire before any DIRECT_CONTRADICTION could be
        returned from Stage 6+.  It is separated from Stage 2 because
        tradeoffs need not involve explicit negation.
        """
        if self._is_tradeoff_pair(label_a, label_b):
            return ClassificationResult(
                contradiction_type="TRADEOFF_RELATION",
                severity=THRESHOLDS["low_severity"],
                confidence=0.88,
                scope="LOCAL",
                explanation=(
                    "Claims describe different aspects of the same subject "
                    "and are simultaneously satisfiable (§18 invariant). "
                    "Both claims may be true; this is a tradeoff, not a "
                    "logical contradiction."
                ),
                stage_reached=5,
                evidence={
                    "tradeoff_aspect_groups": "matched",
                    "cosine_similarity": sim,
                },
            )

        return None

    def _stage_context(
        self,
        label_a: str,
        label_b: str,
        sig_a: Any,
        sig_b: Any,
    ) -> Optional[ClassificationResult]:
        """Stage 6: contextual qualifiers differ → CONTEXTUAL_CONTRADICTION."""
        # Heuristic: if canonical forms share a subject token but differ
        # in predicate family, the contradiction is context-dependent.
        canon_a = sig_a.canonical_form.lower()
        canon_b = sig_b.canonical_form.lower()

        parts_a = canon_a.split("::")
        parts_b = canon_b.split("::")

        # Same subject (index 1 in FAMILY::subject::pred::obj::depthN)
        if len(parts_a) >= 4 and len(parts_b) >= 4:
            subject_a = parts_a[1].strip()
            subject_b = parts_b[1].strip()
            family_a = parts_a[0].strip()
            family_b = parts_b[0].strip()

            if subject_a == subject_b and family_a != family_b:
                return ClassificationResult(
                    contradiction_type="CONTEXTUAL_CONTRADICTION",
                    severity=THRESHOLDS["medium_severity"],
                    confidence=0.65,
                    scope="LOCAL",
                    explanation=(
                        f"Same subject ('{subject_a}') but different predicate "
                        f"families ({family_a} vs {family_b}) — contextual "
                        f"contradiction."
                    ),
                    stage_reached=6,
                    evidence={
                        "shared_subject": subject_a,
                        "family_a": family_a,
                        "family_b": family_b,
                    },
                )

        return None

    def _stage_probabilistic(
        self, node_a: Any, node_b: Any
    ) -> Optional[ClassificationResult]:
        """Stage 7: low-confidence nodes → PROBABILISTIC_DISAGREEMENT."""
        conf_a = node_a.confidence_state.overall_confidence
        conf_b = node_b.confidence_state.overall_confidence

        if conf_a < THRESHOLDS["prob_confidence_cap"] or \
                conf_b < THRESHOLDS["prob_confidence_cap"]:
            return ClassificationResult(
                contradiction_type="PROBABILISTIC_DISAGREEMENT",
                severity=THRESHOLDS["low_severity"],
                confidence=0.60,
                scope="LOCAL",
                explanation=(
                    f"At least one claim has low confidence "
                    f"(a={conf_a:.2f}, b={conf_b:.2f} < "
                    f"{THRESHOLDS['prob_confidence_cap']}). "
                    f"Conflict is probabilistic, not definitive."
                ),
                stage_reached=7,
                evidence={"confidence_a": conf_a, "confidence_b": conf_b},
            )

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_tradeoff_pair(self, label_a: str, label_b: str) -> bool:
        """Return True if (label_a, label_b) is a simultaneously-satisfiable pair.

        Algorithm:
            Both claims must match aspect keywords from DIFFERENT groups
            in _TRADEOFF_ASPECT_GROUPS.  Same-group matches (two cognitive
            benefit claims) are NOT tradeoffs.
        """
        def aspect_groups_for(text: str) -> set:
            return {
                i
                for i, group in enumerate(_TRADEOFF_ASPECT_GROUPS)
                if any(kw in text for kw in group)
            }

        groups_a = aspect_groups_for(label_a)
        groups_b = aspect_groups_for(label_b)

        # They must each match at least one group AND those groups must differ.
        return bool(groups_a) and bool(groups_b) and groups_a != groups_b

    @staticmethod
    def _cosine_sim(
        vec_a: List[float], vec_b: List[float]
    ) -> Optional[float]:
        """Cosine similarity between two embedding vectors.

        Returns None if either vector is empty.
        """
        if not vec_a or not vec_b:
            return None
        try:
            import numpy as np
            a = np.array(vec_a, dtype=float)
            b = np.array(vec_b, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom < 1e-10:
                return None
            return float(np.dot(a, b) / denom)
        except Exception:
            return None
