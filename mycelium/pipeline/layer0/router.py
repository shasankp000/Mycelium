from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mycelium.core.types import Layer0Result
from mycelium.pipeline.layer0.manipulation_detector import ManipulationDetector
from mycelium.pipeline.layer0.objectivity_classifier import ObjectivityClassifier
from mycelium.pipeline.layer0.value_assumption_extractor import ValueAssumptionExtractor

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

REFUSE_CONFIDENCE_THRESHOLD = 0.65

# Hard-refuse labels: only COERCIVE and JAILBREAK_ATTEMPT warrant a hard block.
# LOADED_QUESTION and PRESUPPOSITION_INJECTION get MULTI_PERSPECTIVE instead.
_HARD_REFUSE_LABELS = {"JAILBREAK_ATTEMPT", "COERCIVE"}

# Objectivity labels that map to MULTI_PERSPECTIVE, but only when there is
# genuine value-ladenness — not just causal/evaluative vocabulary in
# an otherwise factual science question.
_VALUE_LADEN_OBJ_LABELS = {"VALUE_LADEN", "SUBJECTIVE"}


class QuestionRouter:
    """
    Layer 0 router.

    Routing priority (highest to lowest):
    1. Hard REFUSE — JAILBREAK_ATTEMPT / COERCIVE AND confidence >= 0.65
    2. DST fusion result — when decisive (dominant BetP >= 0.45)
    3. Rule-based fallback — when DST is unavailable or non-decisive

    Rule-based routing logic:
    — REFUSE:            hard-refuse label + confidence >= threshold
    — MULTI_PERSPECTIVE: LOADED_QUESTION, PRESUPPOSITION_INJECTION,
                         or objectivity = VALUE_LADEN/SUBJECTIVE with
                         no conflicting NOT_MANIPULATIVE + clean signal
    — CLARIFICATION:     objectivity = AMBIGUOUS
    — REASONING_PIPELINE: all clear

    Factual-question guard:
        If the objectivity classifier returns VALUE_LADEN but the manipulation
        classifier returned NOT_MANIPULATIVE with zero coercive/manipulation
        signals, the question is almost certainly a factual science/history
        question that happens to contain causal vocabulary. In this case the
        manipulation result takes precedence and the question is routed to
        REASONING_PIPELINE, not MULTI_PERSPECTIVE.
    """

    def __init__(self) -> None:
        self._manipulation = ManipulationDetector()
        self._objectivity  = ObjectivityClassifier()
        self._assumptions  = ValueAssumptionExtractor()

    def route(self, question: str) -> Layer0Result:
        manip             = self._manipulation.detect(question)
        manip_proba, manip_labels = self._get_manip_proba(question)

        obj               = self._objectivity.classify(question)
        obj_proba, obj_labels = self._get_obj_proba(question)

        assumption_result = self._assumptions.extract(question)
        assumption_strings = assumption_result.assumption_strings
        n_assumptions = len(assumption_result.assumptions)

        dst_meta = {}
        belief = self._dst_fuse(
            manip_proba, manip_labels,
            obj_proba,   obj_labels,
            n_assumptions,
        )
        if belief is not None:
            dst_meta = {
                "dst_pignistic":  belief.pignistic_probs,
                "dst_conflict_k": belief.conflict_k,
                "dst_uncertain":  belief.is_uncertain,
                "dst_sources":    belief.sources,
            }

        if belief is not None and not belief.is_uncertain:
            route = belief.dominant_route
        else:
            route = self._rule_based_route(manip, obj)

        return self._build_result(route, manip, obj, assumption_strings, dst_meta)

    # ------------------------------------------------------------------
    # Raw proba extraction
    # ------------------------------------------------------------------

    def _get_manip_proba(
        self, text: str
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        try:
            from mycelium.pipeline.model_registry import get_layer0_classifier, get_model
            from mycelium.pipeline.layer0.train_layer0_models import _rule_signal_vector
            from mycelium.pipeline.layer0.nlp_preprocessor import get_preprocessor
            artifact = get_layer0_classifier("manipulation_classifier")
            if artifact is None:
                return None, None
            encoder = get_model(
                "sentence-transformers/all-MiniLM-L6-v2",
                model_type="sentence_transformer",
                device="cpu",
            )
            pre  = get_preprocessor()
            _    = pre.analyse(text)
            emb  = encoder.encode([text], convert_to_numpy=True)
            rvec = _rule_signal_vector(text).reshape(1, -1)
            X    = np.hstack([emb, rvec])
            clf  = artifact["model"]
            le   = artifact["label_encoder"]
            proba = clf.predict_proba(X)[0]
            return proba, list(le.classes_)
        except Exception:
            return None, None

    def _get_obj_proba(
        self, text: str
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        try:
            from mycelium.pipeline.model_registry import get_layer0_classifier, get_model
            from mycelium.pipeline.layer0.train_layer0_models import _rule_signal_vector
            artifact = get_layer0_classifier("objectivity_classifier")
            if artifact is None:
                return None, None
            encoder = get_model(
                "sentence-transformers/all-MiniLM-L6-v2",
                model_type="sentence_transformer",
                device="cpu",
            )
            emb  = encoder.encode([text], convert_to_numpy=True)
            rvec = _rule_signal_vector(text).reshape(1, -1)
            X    = np.hstack([emb, rvec])
            clf  = artifact["model"]
            le   = artifact["label_encoder"]
            proba = clf.predict_proba(X)[0]
            return proba, list(le.classes_)
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # DST fusion
    # ------------------------------------------------------------------

    def _dst_fuse(self, manip_proba, manip_labels, obj_proba, obj_labels, n_assumptions: int):
        try:
            from mycelium.pipeline.layer0.dst_fusion import fuse_layer0_classifiers
            return fuse_layer0_classifiers(
                manip_proba=manip_proba,
                manip_labels=manip_labels,
                obj_proba=obj_proba,
                obj_labels=obj_labels,
                n_assumptions=n_assumptions,
                refuse_threshold=REFUSE_CONFIDENCE_THRESHOLD,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Rule-based routing fallback
    # ------------------------------------------------------------------

    def _rule_based_route(self, manip, obj) -> str:
        """
        Conservative rule-based routing used when DST is unavailable or
        non-decisive.

        Branch order is intentional — the confidence gate on REFUSE is
        checked first. The factual-question guard prevents science/history
        questions from being routed to MULTI_PERSPECTIVE purely because
        the objectivity classifier sees causal vocabulary.
        """
        # 1. Hard REFUSE: only high-confidence hard-refuse labels.
        if manip.label in _HARD_REFUSE_LABELS and manip.confidence >= REFUSE_CONFIDENCE_THRESHOLD:
            return "REFUSE"

        # 2. Manipulation-driven MULTI_PERSPECTIVE: loaded or presupposition
        #    questions get a balanced answer regardless of objectivity label.
        if manip.label in {"LOADED_QUESTION", "PRESUPPOSITION_INJECTION"}:
            return "MULTI_PERSPECTIVE"

        # 3. Below-threshold hard-refuse: treat as ambiguous.
        if manip.label in _HARD_REFUSE_LABELS:
            return "MULTI_PERSPECTIVE"

        # 4. Objectivity-driven routing with factual-question guard.
        #    If the objectivity classifier says VALUE_LADEN or SUBJECTIVE,
        #    but the manipulation classifier returned NOT_MANIPULATIVE with
        #    no coercive signals at all, this is almost certainly a factual
        #    question that contains evaluative-sounding words (e.g. "why does
        #    gravity work?" or "explain ferromagnetism").  Route to
        #    REASONING_PIPELINE instead of MULTI_PERSPECTIVE.
        if obj.question_type in _VALUE_LADEN_OBJ_LABELS:
            if manip.label == "NOT_MANIPULATIVE" and not manip.matched_patterns:
                # No manipulation signals at all — trust the factual-question
                # interpretation from the manipulation classifier.
                return "REASONING_PIPELINE"
            return "MULTI_PERSPECTIVE"

        if obj.question_type == "AMBIGUOUS":
            return "CLARIFICATION"

        return "REASONING_PIPELINE"

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(
        self, route: str, manip, obj, assumption_strings, dst_meta: dict
    ) -> Layer0Result:
        if route == "REFUSE":
            return Layer0Result(
                route="REFUSE",
                response_type="REFUSAL",
                metadata={
                    "matched_patterns":   manip.matched_patterns,
                    "manipulation_label": manip.label,
                    "rule_score":         manip.rule_score,
                    "llm_used":           manip.llm_used,
                    "confidence":         manip.confidence,
                    **dst_meta,
                },
            )

        if route == "MULTI_PERSPECTIVE":
            meta = {
                "assumptions":         assumption_strings,
                "objectivity_signals": obj.signals,
                "llm_used":            obj.llm_used,
                **dst_meta,
            }
            if manip.label not in {"NOT_MANIPULATIVE"}:
                meta["manipulation_label"]      = manip.label
                meta["manipulation_confidence"] = manip.confidence
            return Layer0Result(
                route="MULTI_PERSPECTIVE",
                response_type="MULTIPLE_TRUTHS",
                metadata=meta,
            )

        if route == "CLARIFICATION":
            return Layer0Result(
                route="CLARIFICATION",
                response_type="REFRAME_REQUEST",
                metadata={
                    "confidence":          obj.confidence,
                    "objectivity_signals": obj.signals,
                    **dst_meta,
                },
            )

        return Layer0Result(
            route="REASONING_PIPELINE",
            response_type="DEFINITIVE_ANSWER",
            metadata={
                "confidence":          obj.confidence,
                "objectivity_signals": obj.signals,
                "assumptions":         assumption_strings,
                **dst_meta,
            },
        )
