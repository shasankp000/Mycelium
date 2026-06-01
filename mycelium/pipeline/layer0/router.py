from mycelium.core.types import Layer0Result
from mycelium.pipeline.layer0.manipulation_detector import ManipulationDetector
from mycelium.pipeline.layer0.objectivity_classifier import ObjectivityClassifier
from mycelium.pipeline.layer0.value_assumption_extractor import ValueAssumptionExtractor

# Minimum confidence required to hard-REFUSE a request.
# Below this threshold the manipulation signal is treated as ambiguous and
# the request falls through to the objectivity classifier for a nuanced route.
# Rationale: the Layer 0 sklearn models are trained on a relatively small
# corpus; a low-confidence LOADED_QUESTION hit on a clean physics question
# ("Is it possible to create a thermoelectric device?") must never silence
# an otherwise legitimate request.
REFUSE_CONFIDENCE_THRESHOLD = 0.65

# Labels that warrant a hard REFUSE when confidence >= threshold.
# LOADED_QUESTION is intentionally excluded: a loaded question deserves a
# balanced multi-perspective answer, not a hard block.
_HARD_REFUSE_LABELS = {"JAILBREAK_ATTEMPT", "COERCIVE"}


class QuestionRouter:
    """Layer 0 router.

    Runs the three NLP-backed components in sequence and routes to the
    appropriate downstream pipeline stage.

    Routing logic:

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

    metadata keys:
        REFUSE           — {matched_patterns, manipulation_label, rule_score,
                           llm_used, confidence}
        MULTI_PERSPECTIVE — {assumptions, objectivity_signals, llm_used,
                            manipulation_label (if from manipulation path)}
        CLARIFICATION    — {confidence, objectivity_signals}
        REASONING_PIPELINE — {confidence, objectivity_signals,
                             assumptions}
    """

    def __init__(self) -> None:
        self._manipulation = ManipulationDetector()
        self._objectivity = ObjectivityClassifier()
        self._assumptions = ValueAssumptionExtractor()

    def route(self, question: str) -> Layer0Result:
        manip = self._manipulation.detect(question)

        if manip.is_manipulative:
            label = manip.label
            confidence = manip.confidence

            # --- Hard REFUSE: only for high-confidence coercive / jailbreak ---
            if label in _HARD_REFUSE_LABELS and confidence >= REFUSE_CONFIDENCE_THRESHOLD:
                return Layer0Result(
                    route="REFUSE",
                    response_type="REFUSAL",
                    metadata={
                        "matched_patterns": manip.matched_patterns,
                        "manipulation_label": label,
                        "rule_score": manip.rule_score,
                        "llm_used": manip.llm_used,
                        "confidence": confidence,
                    },
                )

            # --- Soft path: LOADED_QUESTION or low-confidence hit ---
            # Route to MULTI_PERSPECTIVE so the question gets a balanced answer
            # rather than being silently dropped.
            obj = self._objectivity.classify(question)
            assumption_result = self._assumptions.extract(question)
            assumption_strings = assumption_result.assumption_strings
            return Layer0Result(
                route="MULTI_PERSPECTIVE",
                response_type="MULTIPLE_TRUTHS",
                metadata={
                    "assumptions": assumption_strings,
                    "objectivity_signals": obj.signals,
                    "manipulation_label": label,
                    "manipulation_confidence": confidence,
                    "llm_used": obj.llm_used or assumption_result.llm_used,
                },
            )

        # --- No manipulation signal: standard objectivity routing ---
        obj = self._objectivity.classify(question)
        assumption_result = self._assumptions.extract(question)
        assumption_strings = assumption_result.assumption_strings

        if obj.question_type == "VALUE_LADEN":
            return Layer0Result(
                route="MULTI_PERSPECTIVE",
                response_type="MULTIPLE_TRUTHS",
                metadata={
                    "assumptions": assumption_strings,
                    "objectivity_signals": obj.signals,
                    "llm_used": obj.llm_used or assumption_result.llm_used,
                },
            )
        if obj.question_type == "AMBIGUOUS":
            return Layer0Result(
                route="CLARIFICATION",
                response_type="REFRAME_REQUEST",
                metadata={
                    "confidence": obj.confidence,
                    "objectivity_signals": obj.signals,
                },
            )
        return Layer0Result(
            route="REASONING_PIPELINE",
            response_type="DEFINITIVE_ANSWER",
            metadata={
                "confidence": obj.confidence,
                "objectivity_signals": obj.signals,
                "assumptions": assumption_strings,
            },
        )
