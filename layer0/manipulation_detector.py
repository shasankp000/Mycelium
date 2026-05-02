from dataclasses import dataclass
from typing import List


@dataclass
class ManipulationDetectionResult:
    is_manipulative: bool
    matched_patterns: List[str]


class ManipulationDetector:
    def __init__(self) -> None:
        self._patterns = {
            "ignore instructions": "instruction_override",
            "do not mention": "concealment_request",
            "you must agree": "forced_agreement",
            "only say yes": "forced_output",
            "bypass safety": "safety_bypass",
        }

    def detect(self, question: str) -> ManipulationDetectionResult:
        text = (question or "").lower()
        matched = [label for phrase, label in self._patterns.items() if phrase in text]
        return ManipulationDetectionResult(is_manipulative=bool(matched), matched_patterns=matched)
