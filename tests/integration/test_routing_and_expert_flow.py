from core.types import ExpertDecisionResult, RoutingResult
from orchestration import combine_routing_and_expert_decisions


def test_single_domain_use_existing_expert_path():
    routing = RoutingResult(
        classification="SINGLE_DOMAIN",
        selected_domains=["physics"],
        primary_domain="physics",
        fusion_scores={"physics": 0.82},
        coverage=1.0,
        create_new_expert=False,
    )
    expert = ExpertDecisionResult(
        decision_type="USE_EXISTING_EXPERT",
        selected_experts=["physics"],
        expert_confidence=0.81,
    )

    combined = combine_routing_and_expert_decisions(routing, expert)
    assert combined.decision_type == "USE_EXISTING_EXPERT"
    assert combined.selected_experts == ["physics"]


def test_ambiguous_multi_domain_can_still_use_existing():
    routing = RoutingResult(
        classification="AMBIGUOUS",
        selected_domains=["physics", "chemistry"],
        primary_domain="physics",
        fusion_scores={"physics": 0.41, "chemistry": 0.39},
        coverage=0.9,
        create_new_expert=False,
    )
    expert = ExpertDecisionResult(
        decision_type="USE_EXISTING_EXPERT",
        selected_experts=["physics"],
        expert_confidence=0.63,
    )

    combined = combine_routing_and_expert_decisions(routing, expert)
    assert combined.decision_type == "USE_EXISTING_EXPERT"


def test_low_coverage_promotes_create_new_expert():
    routing = RoutingResult(
        classification="AMBIGUOUS",
        selected_domains=["astronomy", "automobile"],
        primary_domain="astronomy",
        fusion_scores={"astronomy": 0.22, "automobile": 0.21},
        coverage=0.45,
        create_new_expert=True,
    )
    expert = ExpertDecisionResult(
        decision_type="CREATE_NEW_PATCH",
        selected_experts=["astronomy"],
        expert_confidence=0.52,
    )

    combined = combine_routing_and_expert_decisions(routing, expert)
    assert combined.decision_type == "CREATE_NEW_EXPERT"
