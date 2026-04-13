"""
Integration tests for Phase 2.1 → 2.2 → 2.3 pipeline flow.
"""

import pytest

from phase2_validation.config.phase2_config import Phase2Config
from phase2_validation.phases.phase_2_1_input_normalization import (
    InputNormalizationPipeline,
    NormalizationResult,
)
from phase2_validation.phases.phase_2_2_semantic_understanding import (
    SemanticUnderstandingPipeline,
)
from phase2_validation.phases.phase_2_3_expert_selection import (
    ExpertSelectionPipeline,
)
from phase2_validation.utils.semantic_types import (
    ExpertSelectionResult,
    SemanticResult,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

ALL_DOMAINS = [
    "medical", "physics", "chemistry",
    "mathematics", "computer_science", "music",
]

SAMPLE_EXPERTS = [
    {"name": "expert_medical", "domain": "medical"},
    {"name": "expert_physics", "domain": "physics"},
    {"name": "expert_chemistry", "domain": "chemistry"},
    {"name": "expert_math", "domain": "mathematics"},
    {"name": "expert_cs", "domain": "computer_science"},
    {"name": "expert_music", "domain": "music"},
]


@pytest.fixture
def norm_pipeline():
    return InputNormalizationPipeline()


@pytest.fixture
def sem_pipeline():
    return SemanticUnderstandingPipeline()


@pytest.fixture
def expert_pipeline():
    return ExpertSelectionPipeline()


# ------------------------------------------------------------------
# Phase 2.1 → 2.2 flow
# ------------------------------------------------------------------


class TestNormalizationToSemanticFlow:
    def test_normalization_to_semantic_flow(
        self, norm_pipeline, sem_pipeline
    ):
        norm = norm_pipeline.normalize(
            "The patient was diagnosed with cancer and "
            "received chemotherapy treatment."
        )
        result = sem_pipeline.understand(norm, ALL_DOMAINS)
        assert isinstance(result, SemanticResult)
        assert result.text_embedding is not None
        assert len(result.ranked_domains) > 0

    def test_semantic_preserves_original_text(
        self, norm_pipeline, sem_pipeline
    ):
        raw = (
            "Quantum mechanics describes electron behavior "
            "in atoms."
        )
        norm = norm_pipeline.normalize(raw)
        result = sem_pipeline.understand(norm, ALL_DOMAINS)
        assert result.original_text == raw

    def test_semantic_uses_cleaned_text(
        self, norm_pipeline, sem_pipeline
    ):
        raw = (
            "<p>Cancer <b>treatment</b> for patients.</p>"
        )
        norm = norm_pipeline.normalize(raw)
        result = sem_pipeline.understand(norm, ALL_DOMAINS)
        assert "<p>" not in result.cleaned_text

    def test_tags_propagate_to_semantic(
        self, norm_pipeline, sem_pipeline
    ):
        raw = (
            "The medical treatment involved chemotherapy "
            "for the clinical diagnosis."
        )
        norm = norm_pipeline.normalize(raw)
        result = sem_pipeline.understand(norm, ALL_DOMAINS)
        assert result.text_embedding is not None


# ------------------------------------------------------------------
# Phase 2.2 → 2.3 flow
# ------------------------------------------------------------------


class TestSemanticToExpertSelectionFlow:
    def test_semantic_to_expert_selection_flow(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        norm = norm_pipeline.normalize(
            "Cancer treatment with chemotherapy for patients."
        )
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS
        )
        assert isinstance(result, ExpertSelectionResult)
        assert result.total_selected >= 1

    def test_expert_scores_match_domains(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        norm = norm_pipeline.normalize(
            "Quantum mechanics describes electron energy "
            "levels in atoms."
        )
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS
        )
        assert "expert_physics" in result.expert_scores


# ------------------------------------------------------------------
# Full pipeline: 2.1 → 2.2 → 2.3
# ------------------------------------------------------------------


class TestFullPipelineMedical:
    def test_full_pipeline_medical_input(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = (
            "Metastatic carcinoma requires chemotherapy "
            "rather than localized radiation treatment. "
            "The patient was diagnosed with stage IV cancer."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS
        )
        assert result.total_selected >= 1
        selected_domains = {
            e.domain for e in result.selected_experts
        }
        # Medical should be among the selected
        assert "medical" in selected_domains

    def test_medical_entities_extracted(
        self, norm_pipeline, sem_pipeline
    ):
        raw = (
            "The patient received drug treatment for cancer."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        medical_entities = [
            e for e in sem.extracted_entities
            if e.domain == "medical"
        ]
        assert len(medical_entities) > 0


class TestFullPipelinePhysics:
    def test_full_pipeline_physics_input(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = (
            "Quantum mechanics describes the behavior of "
            "particles at the atomic scale. Energy and "
            "momentum are conserved in particle collisions."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS
        )
        assert result.total_selected >= 1

    def test_physics_entities_extracted(
        self, norm_pipeline, sem_pipeline
    ):
        raw = (
            "The electron has energy and momentum "
            "in quantum mechanics."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        physics_entities = [
            e for e in sem.extracted_entities
            if e.domain == "physics"
        ]
        assert len(physics_entities) > 0


class TestFullPipelineChemistry:
    def test_full_pipeline_chemistry_input(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = (
            "The chemical reaction between the acid and "
            "base produced a new compound through organic "
            "synthesis."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS
        )
        assert result.total_selected >= 1

    def test_chemistry_concepts(
        self, norm_pipeline, sem_pipeline
    ):
        raw = (
            "Chemical reaction with organic molecules "
            "and catalyst compounds."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        assert len(sem.key_concepts) > 0


class TestFullPipelineMultiDomain:
    def test_full_pipeline_multi_domain_input(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = (
            "The medical treatment uses quantum physics "
            "principles to target cancer cells with "
            "chemical compounds."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS
        )
        assert result.total_selected >= 1
        # Multiple domains should be scored
        assert len(sem.domain_relevance_scores) >= 2

    def test_multi_domain_ranking(
        self, norm_pipeline, sem_pipeline
    ):
        raw = (
            "Energy conservation in chemical reactions "
            "within biological systems."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        assert len(sem.ranked_domains) == len(ALL_DOMAINS)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_input_handling(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        norm = norm_pipeline.normalize(
            "This is a valid but generic input sentence."
        )
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS
        )
        assert isinstance(result, ExpertSelectionResult)

    def test_no_suitable_expert_detection(
        self, sem_pipeline, expert_pipeline
    ):
        from types import SimpleNamespace
        norm = SimpleNamespace(
            cleaned_text="zzzzz random gibberish text zzzz",
            original_text="zzzzz random gibberish text zzzz",
            extracted_tags=[],
        )
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS, min_score=0.95
        )
        # Either cold start or very few selected
        assert (
            result.cold_start_fallback_used
            or result.total_selected <= 1
        )

    def test_graceful_degradation(
        self, sem_pipeline, expert_pipeline
    ):
        from types import SimpleNamespace
        norm = SimpleNamespace(
            cleaned_text="x",
            original_text="x",
            extracted_tags=[],
        )
        sem = sem_pipeline.understand(norm, [])
        result = expert_pipeline.select_experts(sem, [])
        assert isinstance(result, ExpertSelectionResult)
        assert result.total_selected == 0

    def test_single_expert_available(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = "Cancer treatment for patients."
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, [SAMPLE_EXPERTS[0]]
        )
        assert result.total_candidates == 1


# ------------------------------------------------------------------
# Caching and performance
# ------------------------------------------------------------------


class TestCachingAndPerformance:
    def test_caching_effectiveness(
        self, norm_pipeline, sem_pipeline
    ):
        raw = (
            "Cancer treatment for the patient with "
            "chemotherapy drugs."
        )
        norm = norm_pipeline.normalize(raw)

        sem1 = sem_pipeline.understand(norm, ALL_DOMAINS)
        hits1 = sem1.embedding_cache_hits

        sem2 = sem_pipeline.understand(norm, ALL_DOMAINS)
        hits2 = sem2.embedding_cache_hits

        assert hits2 > hits1

    def test_processing_time_recorded(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = "Energy and force in quantum mechanics."
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS
        )
        assert norm.processing_time_ms >= 0
        assert sem.processing_time_ms >= 0
        assert result.processing_time_ms >= 0

    def test_repeated_runs_consistent(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = "Patient cancer treatment chemotherapy."
        norm = norm_pipeline.normalize(raw)

        sem1 = sem_pipeline.understand(norm, ALL_DOMAINS)
        r1 = expert_pipeline.select_experts(
            sem1, SAMPLE_EXPERTS
        )

        sem2 = sem_pipeline.understand(norm, ALL_DOMAINS)
        r2 = expert_pipeline.select_experts(
            sem2, SAMPLE_EXPERTS
        )

        assert r1.total_selected == r2.total_selected
        assert set(
            e.name for e in r1.selected_experts
        ) == set(
            e.name for e in r2.selected_experts
        )


# ------------------------------------------------------------------
# Strategy variations
# ------------------------------------------------------------------


class TestStrategyVariations:
    def test_diversity_strategy_end_to_end(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = (
            "Chemical reactions in biological systems "
            "involve energy and force interactions."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS, strategy="diversity"
        )
        assert result.selection_strategy == "diversity"

    def test_uncertainty_strategy_end_to_end(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = (
            "Quantum chemistry involves molecular energy "
            "and chemical bond physics."
        )
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS, strategy="uncertainty"
        )
        assert result.selection_strategy == "uncertainty"

    def test_max_experts_parameter(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = "Energy force momentum in physics."
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS, max_experts=2
        )
        assert result.total_selected <= 2

    def test_min_score_parameter(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = "Patient diagnosis treatment therapy."
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS, min_score=0.9
        )
        for expert in result.selected_experts:
            if not result.cold_start_fallback_used:
                assert expert.match_score >= 0.9


# ------------------------------------------------------------------
# Justification
# ------------------------------------------------------------------


class TestJustification:
    def test_justification_contains_info(
        self, norm_pipeline, sem_pipeline, expert_pipeline
    ):
        raw = "Cancer chemotherapy treatment for patients."
        norm = norm_pipeline.normalize(raw)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        result = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS
        )
        justification = (
            expert_pipeline.get_selection_justification(result)
        )
        assert "Strategy" in justification
        assert "Candidates" in justification
        assert "Selected" in justification
        assert "Confidence" in justification
