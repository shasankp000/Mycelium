"""
Tests for Phase 2.3 — Expert Selection.
"""

import pytest

from phase2_validation.phases.phase_2_3_expert_selection import (
    ExpertGrouping,
    ExpertPoolFilter,
    ExpertRankingEngine,
    ExpertSelectionPipeline,
    InvalidStrategyError,
)
from phase2_validation.utils.semantic_types import (
    ExpertSelectionResult,
    RankedExpert,
    SemanticResult,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_semantic_result(
    domain_scores=None,
    concept_scores=None,
    graph_density=0.3,
    confidence=0.7,
):
    """Build a minimal SemanticResult for testing."""
    return SemanticResult(
        cleaned_text="test",
        domain_relevance_scores=domain_scores or {},
        concept_scores=concept_scores or {},
        graph_density=graph_density,
        confidence_score=confidence,
    )


def _make_experts(*domains):
    """Build a list of expert config dicts."""
    return [
        {"name": f"expert_{d}", "domain": d}
        for d in domains
    ]


def _make_ranked(name, domain, score):
    """Build a RankedExpert with defaults."""
    return RankedExpert(
        name=name,
        domain=domain,
        match_score=score,
        confidence_low=max(score - 0.1, 0.0),
        confidence_high=min(score + 0.1, 1.0),
        ranking_factors={"domain": score},
    )


# ------------------------------------------------------------------
# ExpertRankingEngine
# ------------------------------------------------------------------


class TestExpertRankingEngine:
    def setup_method(self):
        self.engine = ExpertRankingEngine()

    def test_expert_ranking_basic(self):
        sem = _make_semantic_result(
            domain_scores={
                "medical": 0.9,
                "physics": 0.3,
            },
            concept_scores={"cancer": 0.8},
        )
        experts = _make_experts("medical", "physics")
        ranked = self.engine.rank_experts(sem, experts)
        assert len(ranked) == 2
        assert ranked[0].match_score >= ranked[1].match_score

    def test_expert_match_scoring(self):
        sem = _make_semantic_result(
            domain_scores={"medical": 0.9},
            concept_scores={"cancer": 0.8},
        )
        expert = {"name": "med_expert", "domain": "medical"}
        score = self.engine.compute_expert_match_score(
            sem, expert
        )
        assert 0.0 <= score <= 1.0

    def test_confidence_interval_computation(self):
        factors = {
            "domain": 0.9,
            "concepts": 0.5,
            "graph": 0.3,
            "confidence": 0.7,
        }
        low, high = self.engine.compute_confidence_interval(
            0.7, factors
        )
        assert low <= 0.7
        assert high >= 0.7
        assert low >= 0.0
        assert high <= 1.0

    def test_confidence_interval_narrow_for_uniform(self):
        factors = {
            "domain": 0.5,
            "concepts": 0.5,
            "graph": 0.5,
            "confidence": 0.5,
        }
        low, high = self.engine.compute_confidence_interval(
            0.5, factors
        )
        assert (high - low) < 0.5

    def test_ranking_preserves_all_experts(self):
        sem = _make_semantic_result(
            domain_scores={"a": 0.1, "b": 0.5, "c": 0.9}
        )
        experts = _make_experts("a", "b", "c")
        ranked = self.engine.rank_experts(sem, experts)
        assert len(ranked) == 3

    def test_ranking_factors_structure(self):
        sem = _make_semantic_result(
            domain_scores={"medical": 0.8},
            concept_scores={"test": 0.5},
        )
        expert = {"name": "e1", "domain": "medical"}
        ranked = self.engine.rank_experts(sem, [expert])
        assert "domain" in ranked[0].ranking_factors
        assert "concepts" in ranked[0].ranking_factors
        assert "graph" in ranked[0].ranking_factors
        assert "confidence" in ranked[0].ranking_factors


# ------------------------------------------------------------------
# ExpertPoolFilter
# ------------------------------------------------------------------


class TestExpertPoolFilter:
    def setup_method(self):
        self.pool_filter = ExpertPoolFilter()

    def test_pool_filtering(self):
        experts = [
            _make_ranked("e1", "medical", 0.9),
            _make_ranked("e2", "physics", 0.05),
        ]
        sem = _make_semantic_result()
        result = self.pool_filter.filter_pool(experts, sem)
        assert len(result["selected"]) == 1
        assert len(result["filtered_out"]) == 1

    def test_threshold_application(self):
        experts = [
            _make_ranked("e1", "medical", 0.9),
            _make_ranked("e2", "physics", 0.5),
            _make_ranked("e3", "chemistry", 0.2),
        ]
        filtered = self.pool_filter.apply_thresholds(
            experts, min_score=0.3, max_pool_size=4
        )
        assert len(filtered) == 2

    def test_max_pool_size(self):
        experts = [
            _make_ranked(f"e{i}", "domain", 0.9)
            for i in range(10)
        ]
        filtered = self.pool_filter.apply_thresholds(
            experts, min_score=0.0, max_pool_size=3
        )
        assert len(filtered) == 3

    def test_detect_no_suitable_expert_empty(self):
        assert self.pool_filter.detect_no_suitable_expert([])

    def test_detect_no_suitable_expert_low_score(self):
        experts = [_make_ranked("e1", "medical", 0.1)]
        assert self.pool_filter.detect_no_suitable_expert(
            experts
        )

    def test_detect_suitable_expert(self):
        experts = [_make_ranked("e1", "medical", 0.8)]
        assert not self.pool_filter.detect_no_suitable_expert(
            experts
        )

    def test_all_filtered_out(self):
        experts = [
            _make_ranked("e1", "medical", 0.01),
        ]
        sem = _make_semantic_result()
        result = self.pool_filter.filter_pool(experts, sem)
        assert len(result["selected"]) == 0
        assert len(result["filtered_out"]) == 1


# ------------------------------------------------------------------
# ExpertGrouping
# ------------------------------------------------------------------


class TestExpertGrouping:
    def setup_method(self):
        self.grouping = ExpertGrouping()

    def test_expert_grouping_by_domain(self):
        experts = [
            _make_ranked("e1", "medical", 0.9),
            _make_ranked("e2", "medical", 0.7),
            _make_ranked("e3", "physics", 0.8),
        ]
        groups = self.grouping.group_by_domain(experts)
        assert "medical" in groups
        assert "physics" in groups
        assert len(groups["medical"]) == 2

    def test_prioritization_greedy(self):
        experts = [
            _make_ranked("e1", "medical", 0.5),
            _make_ranked("e2", "physics", 0.9),
        ]
        ordered = self.grouping.prioritize_experts(
            experts, strategy="greedy"
        )
        assert ordered[0].name == "e2"

    def test_prioritization_diversity(self):
        experts = [
            _make_ranked("e1", "medical", 0.9),
            _make_ranked("e2", "medical", 0.8),
            _make_ranked("e3", "physics", 0.7),
        ]
        ordered = self.grouping.prioritize_experts(
            experts, strategy="diversity"
        )
        # Both domains should appear
        domains = [e.domain for e in ordered]
        assert "medical" in domains
        assert "physics" in domains

    def test_prioritization_uncertainty(self):
        experts = [
            RankedExpert(
                name="e1", domain="medical",
                match_score=0.5,
                confidence_low=0.1,
                confidence_high=0.9,
            ),
            RankedExpert(
                name="e2", domain="physics",
                match_score=0.7,
                confidence_low=0.6,
                confidence_high=0.8,
            ),
        ]
        ordered = self.grouping.prioritize_experts(
            experts, strategy="uncertainty"
        )
        # e1 has wider interval so should be first
        assert ordered[0].name == "e1"

    def test_prioritization_invalid_strategy(self):
        with pytest.raises(InvalidStrategyError):
            self.grouping.prioritize_experts(
                [], strategy="nonexistent"
            )

    def test_diverse_pool_selection(self):
        experts = [
            _make_ranked("e1", "medical", 0.9),
            _make_ranked("e2", "medical", 0.8),
            _make_ranked("e3", "physics", 0.7),
            _make_ranked("e4", "chemistry", 0.6),
        ]
        pool = self.grouping.select_diverse_pool(
            experts, pool_size=3
        )
        assert len(pool) == 3
        domains = {e.domain for e in pool}
        # Should cover at least 2 domains
        assert len(domains) >= 2

    def test_diverse_pool_small_input(self):
        experts = [_make_ranked("e1", "medical", 0.9)]
        pool = self.grouping.select_diverse_pool(
            experts, pool_size=3
        )
        assert len(pool) == 1


# ------------------------------------------------------------------
# ExpertSelectionPipeline
# ------------------------------------------------------------------


class TestExpertSelectionPipeline:
    def setup_method(self):
        self.pipeline = ExpertSelectionPipeline()

    def test_full_selection_pipeline(self):
        sem = _make_semantic_result(
            domain_scores={
                "medical": 0.9,
                "physics": 0.3,
            },
            concept_scores={"cancer": 0.8},
        )
        experts = _make_experts("medical", "physics")
        result = self.pipeline.select_experts(sem, experts)
        assert isinstance(result, ExpertSelectionResult)
        assert result.total_candidates == 2
        assert result.total_selected >= 1

    def test_pipeline_greedy_strategy(self):
        sem = _make_semantic_result(
            domain_scores={"medical": 0.9, "physics": 0.8},
            concept_scores={"test": 0.5},
        )
        experts = _make_experts("medical", "physics")
        result = self.pipeline.select_experts(
            sem, experts, strategy="greedy"
        )
        assert result.selection_strategy == "greedy"

    def test_pipeline_diversity_strategy(self):
        sem = _make_semantic_result(
            domain_scores={
                "medical": 0.9,
                "physics": 0.8,
                "chemistry": 0.7,
            },
            concept_scores={"test": 0.5},
        )
        experts = _make_experts(
            "medical", "physics", "chemistry"
        )
        result = self.pipeline.select_experts(
            sem, experts, strategy="diversity"
        )
        assert result.selection_strategy == "diversity"

    def test_pipeline_cold_start(self):
        sem = _make_semantic_result(
            domain_scores={"unknown": 0.05},
        )
        experts = _make_experts("unknown")
        result = self.pipeline.select_experts(
            sem, experts, min_score=0.5
        )
        assert result.cold_start_fallback_used

    def test_pipeline_invalid_strategy(self):
        sem = _make_semantic_result()
        experts = _make_experts("medical")
        with pytest.raises(InvalidStrategyError):
            self.pipeline.select_experts(
                sem, experts, strategy="bad"
            )

    def test_pipeline_expert_scores(self):
        sem = _make_semantic_result(
            domain_scores={"medical": 0.9},
            concept_scores={"test": 0.5},
        )
        experts = _make_experts("medical")
        result = self.pipeline.select_experts(sem, experts)
        assert "expert_medical" in result.expert_scores

    def test_pipeline_justification(self):
        sem = _make_semantic_result(
            domain_scores={"medical": 0.9},
            concept_scores={"test": 0.5},
        )
        experts = _make_experts("medical")
        result = self.pipeline.select_experts(sem, experts)
        justification = (
            self.pipeline.get_selection_justification(result)
        )
        assert "Strategy" in justification
        assert "Candidates" in justification

    def test_pipeline_processing_time(self):
        sem = _make_semantic_result(
            domain_scores={"medical": 0.9},
        )
        experts = _make_experts("medical")
        result = self.pipeline.select_experts(sem, experts)
        assert result.processing_time_ms >= 0

    def test_pipeline_confidence(self):
        sem = _make_semantic_result(
            domain_scores={"medical": 0.9},
            concept_scores={"test": 0.5},
        )
        experts = _make_experts("medical")
        result = self.pipeline.select_experts(sem, experts)
        assert 0.0 <= result.confidence_in_selection <= 1.0

    def test_pipeline_routing_single_domain(self):
        sem = _make_semantic_result(
            domain_scores={"medical": 0.2, "physics": 0.9},
        )
        experts = _make_experts("medical", "physics")
        routing_context = {
            "classification": "SINGLE_DOMAIN",
            "primary_domain": "medical",
        }
        result = self.pipeline.select_experts(
            sem, experts, routing_context=routing_context
        )
        selected_domains = {e.domain for e in result.selected_experts}
        assert "medical" in selected_domains

    def test_pipeline_routing_multi_domain(self):
        sem = _make_semantic_result(
            domain_scores={"medical": 0.9, "physics": 0.8},
        )
        experts = _make_experts("medical", "physics", "chemistry")
        routing_context = {
            "classification": "MULTI_DOMAIN",
            "candidate_domains": ["physics", "medical"],
        }
        result = self.pipeline.select_experts(
            sem, experts, routing_context=routing_context
        )
        selected_domains = {e.domain for e in result.selected_experts}
        assert selected_domains.issubset({"medical", "physics"})

    def test_pipeline_routing_no_expert(self):
        sem = _make_semantic_result(domain_scores={"medical": 0.9})
        experts = _make_experts("medical")
        routing_context = {"classification": "NO_EXPERT_AVAILABLE"}
        result = self.pipeline.select_experts(
            sem, experts, routing_context=routing_context
        )
        assert result.total_selected == 0
        assert result.cold_start_fallback_used
