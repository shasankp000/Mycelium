from core.types import Layer0Result
from mycelium.pipeline.layer0.manipulation_detector import ManipulationDetector
from mycelium.pipeline.layer0.objectivity_classifier import ObjectivityClassifier
from mycelium.pipeline.layer0.value_assumption_extractor import ValueAssumptionExtractor


class QuestionRouter:
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
                metadata={"matched_patterns": manip.matched_patterns},
            )

        obj = self._objectivity.classify(question)
        assumptions = self._assumptions.extract(question)

        if obj.question_type == "VALUE_LADEN":
            return Layer0Result(
                route="MULTI_PERSPECTIVE",
                response_type="MULTIPLE_TRUTHS",
                metadata={"assumptions": assumptions.assumptions},
            )
        if obj.question_type == "AMBIGUOUS":
            return Layer0Result(
                route="CLARIFICATION",
                response_type="REFRAME_REQUEST",
                metadata={"confidence": obj.confidence},
            )
        return Layer0Result(
            route="REASONING_PIPELINE",
            response_type="DEFINITIVE_ANSWER",
            metadata={"confidence": obj.confidence},
        )
