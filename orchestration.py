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
    - If routing indicates create_new_expert=True AND the expert decision is
      already CREATE_NEW_EXPERT, leave it unchanged.
    - CREATE_NEW_PATCH is intentionally NOT promoted: the expert system
      determined a known domain exists but has no trained model yet, which
      is a lighter-weight situation than a fully unknown domain.  Promoting
      it to CREATE_NEW_EXPERT would trigger unnecessary heavy pipeline work.
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

    routing_create_new_expert = _extract_bool("create_new_expert", routing, default=False)

    # Trip 1 fix: only promote when the expert decision is not already
    # USE_EXISTING_EXPERT *and* not CREATE_NEW_PATCH.
    # CREATE_NEW_PATCH must be preserved — it means a known domain exists
    # but needs a new model, not that the domain is completely unknown.
    if (
        routing_create_new_expert
        and result.decision_type not in ("USE_EXISTING_EXPERT", "CREATE_NEW_PATCH")
    ):
        result = replace(result, decision_type="CREATE_NEW_EXPERT")

    return result
