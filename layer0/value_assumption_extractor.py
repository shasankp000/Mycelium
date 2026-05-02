from dataclasses import dataclass, field
from typing import List


@dataclass
class ValueAssumptionResult:
    assumptions: List[str] = field(default_factory=list)


class ValueAssumptionExtractor:
    def extract(self, question: str) -> ValueAssumptionResult:
        text = (question or "").lower()
        assumptions = []
        if "best" in text:
            assumptions.append("assumes_single_optimal_answer")
        if "should" in text:
            assumptions.append("assumes_normative_judgment")
        if "always" in text or "never" in text:
            assumptions.append("assumes_universal_rule")
        return ValueAssumptionResult(assumptions=assumptions)
