from core.types import ExpertDecisionResult, RoutingResult


def combine_routing_and_expert_decisions(
    routing: RoutingResult,
    expert: ExpertDecisionResult,
) -> ExpertDecisionResult:
    if routing.create_new_expert and expert.decision_type != "USE_EXISTING_EXPERT":
        expert.decision_type = "CREATE_NEW_EXPERT"
    return expert
