from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional


@dataclass
class Layer0Result:
    route: Literal[
        "REASONING_PIPELINE",
        "MULTI_PERSPECTIVE",
        "REFUSE",
        "CLARIFICATION",
    ]
    response_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    classification: str
    selected_domains: List[str] = field(default_factory=list)
    primary_domain: Optional[str] = None
    fusion_scores: Dict[str, float] = field(default_factory=dict)
    coverage: float = 0.0
    create_new_expert: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "classification": self.classification,
            "selected_experts": list(self.selected_domains),
            "candidate_domains": list(self.selected_domains),
            "primary_domain": self.primary_domain,
            "fused_scores": dict(self.fusion_scores),
            "coverage_met": self.coverage >= 1.0,
            "create_new_expert": self.create_new_expert,
        }
        data.update(self.metadata)
        return data

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass
class ExpertDecisionResult:
    decision_type: Literal[
        "USE_EXISTING_EXPERT",
        "CREATE_NEW_PATCH",
        "CREATE_NEW_EXPERT",
    ]
    selected_experts: List[str] = field(default_factory=list)
    expert_confidence: float = 0.0
    ood_penalty: float = 0.0
    is_ood: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
