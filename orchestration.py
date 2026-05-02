from typing import Union, Dict, Any
from dataclasses import replace
from core.types import ExpertDecisionResult, RoutingResult


def _extract_bool(field: str, source: Union[Dict[str, Any], object], default: bool = False) -> bool:
    """Helper to safely extract a boolean attribute or dict key."""
    if isinstance(source, dict):
        return bool(source.get(field, default))
    return bool(getattr(source, field, default))


def combine_routing_and_expert_decisions(
    routing: Union[RoutingResult, Dict[str, Any]],
    expert: Union[ExpertDecisionResult, Dict[str, Any]],
) -> ExpertDecisionResult:
    """Combine routing and expert decisions.

    Both inputs can be either dicts or objects with attributes.
    Always returns an ExpertDecisionResult instance.

    Rules:
    - Normalize expert decision into ExpertDecisionResult.
    - Map CLARIFICATION to CREATE_NEW_EXPERT (for backward compatibility).
    - If routing indicates create_new_expert=True and the expert decision is
      not USE_EXISTING_EXPERT, promote the decision to CREATE_NEW_EXPERT.
    """
    # Normalize expert result
    if isinstance(expert, dict):
        decision_type = expert.get("decision_type", "CREATE_NEW_PATCH")
        selected_experts = expert.get("selected_experts") or []
        expert_confidence = float(expert.get("expert_confidence", 0.0))
        ood_penalty = float(expert.get("ood_penalty", 0.0))
        is_ood = bool(expert.get("is_ood", False))
        metadata = expert.get("metadata") or {}
    else:
        decision_type = getattr(expert, "decision_type", "CREATE_NEW_PATCH")
        selected_experts = getattr(expert, "selected_experts", []) or []
        expert_confidence = getattr(expert, "expert_confidence", 0.0)
        ood_penalty = getattr(expert, "ood_penalty", 0.0)
        is_ood = getattr(expert, "is_ood", False)
        metadata = getattr(expert, "metadata", {}) or {}

    # Backwards compatibility: treat CLARIFICATION as CREATE_NEW_EXPERT
    if decision_type == "CLARIFICATION":
        decision_type = "CREATE_NEW_EXPERT"

    result = ExpertDecisionResult(
        decision_type=decision_type,
        selected_experts=selected_experts,
        expert_confidence=expert_confidence,
        ood_penalty=ood_penalty,
        is_ood=is_ood,
        metadata=metadata,
    )

    # Combine with routing signal for new expert creation
    routing_create_new_expert = _extract_bool("create_new_expert", routing, default=False)

    if routing_create_new_expert and result.decision_type != "USE_EXISTING_EXPERT":
        # Promote to CREATE_NEW_EXPERT when routing signals insufficient coverage
        result = replace(result, decision_type="CREATE_NEW_EXPERT")

    return result
