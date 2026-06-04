from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mycelium.core.types import Layer0Result
from mycelium.pipeline.layer0.manipulation_detector import ManipulationDetector
from mycelium.pipeline.layer0.objectivity_classifier import ObjectivityClassifier
from mycelium.pipeline.layer0.value_assumption_extractor import ValueAssumptionExtractor

# ---------------------------------------------------------------------------
# Thresholds (single source of truth — also imported by manipulation_detector)
# ---------------------------------------------------------------------------

# Minimum confidence required to hard-REFUSE a request.
# Below this threshold the manipulation signal is treated as ambiguous and
# the request falls through to the objectivity classifier for a nuanced route.
REFUSE_CONFIDENCE_THRESHOLD = 0.65

# Labels that warrant a hard REFUSE when confidence >= threshold.
# LOADED_QUESTION is intentionally excluded: a loaded question deserves a
# balanced multi-perspective answer, not a hard block.
_HARD_REFUSE_LABELS = {"JAILBREAK_ATTEMPT", "COERCIVE"}


class QuestionRouter:
    """
    Layer 0 router.

    Runs the three NLP-backed components in sequence and routes to the
    appropriate downstream pipeline stage.

    When trained sklearn models are available the routing decision is
    backed by Dempster-Shafer Theory (DST) fusion of all three
    classifier outputs (see ``layer0/dst_fusion.py``).  On cold-start
    (models not yet trained) the existing rule-based routing logic is
    used as a fallback.

    Routing logic
    -------------
    REFUSE
        — ManipulationDetector fired with a HARD_REFUSE label
          (JAILBREAK_ATTEMPT or COERCIVE) AND confidence >= 0.65.
          Low-confidence hits fall through to the objectivity path.

    MULTI_PERSPECTIVE
        — ObjectivityClassifier returned VALUE_LADEN, OR
        — ManipulationDetector returned LOADED_QUESTION (any confidence)
          or a hard-refuse label below the confidence threshold.
          A "loaded" question still deserves a balanced answer.

    CLARIFICATION
        — ObjectivityClassifier returned AMBIGUOUS

    REASONING_PIPELINE
        — all clear, proceed to deep reasoning

    Metadata keys
    -------------
    REFUSE
        matched_patterns, manipulation_label, rule_score, llm_used,
        confidence, [dst_pignistic, dst_conflict_k, dst_uncertain]
    MULTI_PERSPECTIVE
        assumptions, objectivity_signals, llm_used,
        manipulation_label (if from manipulation path),
        [dst_pignistic, dst_conflict_k, dst_uncertain]
    CLARIFICATION
        confidence, objectivity_signals,
        [dst_pignistic, dst_conflict_k, dst_uncertain]
    REASONING_PIPELINE
        confidence, objectivity_signals, assumptions,
        [dst_pignistic, dst_conflict_k, dst_uncertain]
    """

    def __init__(self) -> None:
        self._manipulation = ManipulationDetector()
        self._objectivity  = ObjectivityClassifier()
        self._assumptions  = ValueAssumptionExtractor()

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def route(self, question: str) -> Layer0Result:
        """Route ``question`` to the appropriate downstream stage."""
        # Run all three classifiers, collecting raw proba vectors when
        # trained models are available.
        manip           = self._manipulation.detect(question)
        manip_proba, manip_labels = self._get_manip_proba(question)

        obj             = self._objectivity.classify(question)
        obj_proba, obj_labels = self._get_obj_proba(question)

        assumption_result = self._assumptions.extract(question)
        assumption_strings = assumption_result.assumption_strings
        n_assumptions = len(assumption_result.assumptions)

        # --- Attempt DST fusion ---
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

        # --- Determine final route ---
        # Use DST dominant_route when available; fall back to rule-based logic.
        if belief is not None and not belief.is_uncertain:
            route = belief.dominant_route
        else:
            route = self._rule_based_route(manip, obj)

        # --- Build Layer0Result ---
        return self._build_result(route, manip, obj, assumption_strings, dst_meta)

    # ------------------------------------------------------------------
    # Raw proba extraction helpers
    # ------------------------------------------------------------------

    def _get_manip_proba(
        self, text: str
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """Return (proba_vector, label_order) from the manipulation model."""
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
            _    = pre.analyse(text)  # warm-up / caching side-effect
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
        """Return (proba_vector, label_order) from the objectivity model."""
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
    # DST fusion wrapper
    # ------------------------------------------------------------------

    def _dst_fuse(
        self,
        manip_proba, manip_labels,
        obj_proba,   obj_labels,
        n_assumptions: int,
    ):
        """Run DST fusion; return Layer0BeliefState or None on any failure."""
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
    # Rule-based routing fallback (unchanged from original)
    # ------------------------------------------------------------------

    def _rule_based_route(self, manip, obj) -> str:
        """Original deterministic routing logic used when DST is unavailable."""
        if manip.is_manipulative:
            return "REFUSE"
        if manip.label in _HARD_REFUSE_LABELS and manip.confidence < REFUSE_CONFIDENCE_THRESHOLD:
            return "MULTI_PERSPECTIVE"
        if manip.label in {"LOADED_QUESTION", "PRESUPPOSITION_INJECTION"}:
            return "MULTI_PERSPECTIVE"
        if obj.question_type == "VALUE_LADEN":
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
        """Construct a Layer0Result for the given route."""
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

        # REASONING_PIPELINE (default)
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
