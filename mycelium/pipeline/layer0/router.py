from mycelium.core.types import Layer0Result
from mycelium.pipeline.layer0.manipulation_detector import ManipulationDetector
from mycelium.pipeline.layer0.objectivity_classifier import ObjectivityClassifier
from mycelium.pipeline.layer0.value_assumption_extractor import ValueAssumptionExtractor


class QuestionRouter:
    """Layer 0 router.

    Runs the three NLP-backed components in sequence and routes to the
    appropriate downstream pipeline stage.

    Routing logic (unchanged from original; signal richness upgraded):

        REFUSE           — ManipulationDetector fired (any non-NOT_MANIPULATIVE label)
        MULTI_PERSPECTIVE — ObjectivityClassifier returned VALUE_LADEN
        CLARIFICATION    — ObjectivityClassifier returned AMBIGUOUS
        REASONING_PIPELINE — all clear, proceed to deep reasoning

    metadata is now structurally richer:
        REFUSE           — {matched_patterns, manipulation_label, rule_score, llm_used}
        MULTI_PERSPECTIVE — {assumptions: List[str], llm_used}
        CLARIFICATION    — {confidence, objectivity_signals}
        REASONING_PIPELINE — {confidence, objectivity_signals, assumptions: List[str]}
    """

    def __init__(self) -> None:
        self._manipulation = ManipulationDetector()
        self._objectivity = ObjectivityClassifier()
        self._assumptions = ValueAssumptionExtractor()

    def route(self, question: str) -> Layer0Result:
        manip = self._manipulation.detect(question)
        if manip.is_manipulative:
            return Layer0Result(
                route="REFUSE",
                response_type="REFUSAL",
                metadata={
                    "matched_patterns": manip.matched_patterns,
                    "manipulation_label": manip.label,
                    "rule_score": manip.rule_score,
                    "llm_used": manip.llm_used,
                },
            )

        obj = self._objectivity.classify(question)
        assumption_result = self._assumptions.extract(question)

        # Use back-compat string list for existing consumers
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
