from dataclasses import dataclass
from typing import Literal


QuestionType = Literal["OBJECTIVE", "VALUE_LADEN", "AMBIGUOUS"]


@dataclass
class ObjectivityResult:
    question_type: QuestionType
    confidence: float


class ObjectivityClassifier:
    def classify(self, question: str) -> ObjectivityResult:
        text = (question or "").lower().strip()
        if not text or len(text.split()) < 3:
            return ObjectivityResult(question_type="AMBIGUOUS", confidence=0.6)

        value_cues = ["best", "should", "moral", "ethical", "right thing", "good or bad"]
        ambiguous_cues = ["this", "that", "it", "something"]

        if any(c in text for c in value_cues):
            return ObjectivityResult(question_type="VALUE_LADEN", confidence=0.8)
        if text.endswith("?") and sum(1 for c in ambiguous_cues if c in text) >= 2:
            return ObjectivityResult(question_type="AMBIGUOUS", confidence=0.7)
        return ObjectivityResult(question_type="OBJECTIVE", confidence=0.75)
