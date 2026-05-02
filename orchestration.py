from typing import Union, Dict, Any
from dataclasses import replace
from core.types import ExpertDecisionResult, RoutingResult

def combine_routing_and_expert_decisions(
    routing: Union[RoutingResult, Dict[str, Any]],
    expert: Union[ExpertDecisionResult, Dict[str, Any]],
) -> ExpertDecisionResult:
    """Combine routing and expert decisions.
    
    Both can be either a dict or an object with attributes.
    Always returns an ExpertDecisionResult object.
    """
    # Get routing route info
    routing_route = routing.get("route") if isinstance(routing, dict) else getattr(routing, "route", None)
    
    # Get expert fields from dict or object
    expert_decisions = expert.get("decision_type") if isinstance(expert, dict) else getattr(expert, "decision_type", "CREATE_NEW_PATCH")
    expert_decisions = "CREATE_NEW_EXPERT" if expert_decisions == "CLARIFICATION" else expert_decisions
    
    selected_experts = expert.get("selected_experts") if isinstance(expert, dict) else getattr(expert, "selected_experts", [])
    expert_confidence = float(expert.get("expert_confidence", 0.0)) if isinstance(expert, dict) else getattr(expert, "expert_confidence", 0.0)
    ood_penalty = float(expert.get("ood_penalty", 0.0)) if isinstance(expert, dict) else getattr(expert, "ood_penalty", 0.0)
    is_ood = bool(expert.get("is_ood", False)) if isinstance(expert, dict) else getattr(expert, "is_ood", False)
    metadata = expert.get("metadata") if isinstance(expert, dict) and expert.get("metadata") else getattr(expert, "metadata", {})
    
    return ExpertDecisionResult(
        decision_type=expert_decisions,
        selected_experts=selected_experts or [],
        expert_confidence=expert_confidence,
        ood_penalty=ood_penalty,
        is_ood=is_ood,
        metadata=metadata or {},
    )
