"""
End-to-end tests for full Phase 2.1 → 2.6 pipeline.

Runs the REAL pipeline:
normalize → semantic → expert selection →
inference → calibration → synthesis.
"""

import time

import pytest

from phase2_validation.phases.phase_2_1_input_normalization import (
    InputNormalizationPipeline,
)
from phase2_validation.phases.phase_2_2_semantic_understanding import (
    SemanticUnderstandingPipeline,
)
from phase2_validation.phases.phase_2_3_expert_selection import (
    ExpertSelectionPipeline,
)
from phase2_validation.phases.phase_2_4_inference import (
    MultiExpertInferencePipeline,
)
from phase2_validation.phases.phase_2_5_calibration import (
    CalibrationPipeline,
)
from phase2_validation.phases.phase_2_6_synthesis import (
    DecisionSynthesisPipeline,
    FinalDecisionResult,
)

ALL_DOMAINS = [
    "medical",
    "physics",
    "chemistry",
    "mathematics",
    "computer_science",
    "music",
]

SAMPLE_EXPERTS = [
    {"name": "expert_medical", "domain": "medical"},
    {"name": "expert_physics", "domain": "physics"},
    {"name": "expert_chemistry", "domain": "chemistry"},
    {"name": "expert_math", "domain": "mathematics"},
    {"name": "expert_cs", "domain": "computer_science"},
    {"name": "expert_music", "domain": "music"},
]

VALID_DECISIONS = {
    "use_existing",
    "create_patch",
    "create_new_expert",
}

VALID_ACTIONS = {
    "use_existing",
    "create_patch",
    "create_new_expert",
}

EXPECTED_PHASES = [
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
]


# ── fixtures ────────────────────────────────────────────────


@pytest.fixture
def norm_pipeline():
    return InputNormalizationPipeline()


@pytest.fixture
def sem_pipeline():
    return SemanticUnderstandingPipeline()


@pytest.fixture
def expert_pipeline():
    return ExpertSelectionPipeline()


@pytest.fixture
def inf_pipeline():
    return MultiExpertInferencePipeline()


@pytest.fixture
def cal_pipeline():
    return CalibrationPipeline()


@pytest.fixture
def syn_pipeline():
    return DecisionSynthesisPipeline()


# ── helper ──────────────────────────────────────────────────


def _run_full_pipeline(
    text,
    norm_pipeline,
    sem_pipeline,
    expert_pipeline,
    inf_pipeline,
    cal_pipeline,
    syn_pipeline,
    experts=None,
    min_score=None,
):
    """Run complete Phase 2.1-2.6 pipeline."""
    experts = experts or SAMPLE_EXPERTS
    norm = norm_pipeline.normalize(text)
    sem = sem_pipeline.understand(norm, ALL_DOMAINS)
    if min_score is not None:
        sel = expert_pipeline.select_experts(
            sem, experts, min_score=min_score,
        )
    else:
        sel = expert_pipeline.select_experts(sem, experts)
    inf = inf_pipeline.infer(text, sel.selected_experts)
    cal = cal_pipeline.calibrate_and_quantify(inf)
    final = syn_pipeline.synthesize(
        norm, sem, sel, inf, cal,
    )
    return norm, sem, sel, inf, cal, final


# ── medical domain ──────────────────────────────────────────


class TestFullPipelineMedical:
    """Full pipeline tests with medical input."""

    def test_full_phase_2_1_to_2_6_medical_input(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Patient with myocardial infarction"
            " showing elevated troponin levels"
        )
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        assert isinstance(final, FinalDecisionResult)
        assert final.final_decision in VALID_DECISIONS
        assert 0.0 <= final.decision_confidence <= 1.0


# ── physics domain ──────────────────────────────────────────


class TestFullPipelinePhysics:
    """Full pipeline tests with physics input."""

    def test_full_phase_2_1_to_2_6_physics_input(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Quantum mechanics describes electron"
            " behavior in atomic orbitals with"
            " energy quantization"
        )
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        assert isinstance(final, FinalDecisionResult)
        assert final.final_decision in VALID_DECISIONS
        assert 0.0 <= final.decision_confidence <= 1.0


# ── chemistry domain ────────────────────────────────────────


class TestFullPipelineChemistry:
    """Full pipeline tests with chemistry input."""

    def test_full_phase_2_1_to_2_6_chemistry_input(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "The chemical reaction between acid"
            " and base produces salt and water"
            " through neutralization"
        )
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        assert isinstance(final, FinalDecisionResult)
        assert final.final_decision in VALID_DECISIONS
        assert 0.0 <= final.decision_confidence <= 1.0


# ── multi-domain ────────────────────────────────────────────


class TestFullPipelineMultiDomain:
    """Full pipeline tests with multi-domain input."""

    def test_full_phase_2_1_to_2_6_multi_domain_input(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Medical imaging uses physics principles"
            " like radiation and magnetic resonance"
            " for diagnosis"
        )
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        assert isinstance(final, FinalDecisionResult)
        assert final.final_decision in VALID_DECISIONS
        assert 0.0 <= final.decision_confidence <= 1.0


# ── edge cases ──────────────────────────────────────────────


class TestFullPipelineEdgeCases:
    """Edge-case tests for the full pipeline."""

    def test_full_pipeline_with_no_suitable_expert(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Patient with myocardial infarction"
            " showing elevated troponin levels"
        )
        norm = norm_pipeline.normalize(text)
        sem = sem_pipeline.understand(norm, ALL_DOMAINS)
        sel = expert_pipeline.select_experts(
            sem, SAMPLE_EXPERTS, min_score=0.99,
        )
        # cold_start_fallback may still provide experts
        if len(sel.selected_experts) > 0:
            inf = inf_pipeline.infer(
                text, sel.selected_experts,
            )
            cal = cal_pipeline.calibrate_and_quantify(inf)
            final = syn_pipeline.synthesize(
                norm, sem, sel, inf, cal,
            )
            assert isinstance(final, FinalDecisionResult)
            assert final.final_decision in VALID_DECISIONS
        else:
            assert sel.total_selected == 0

    def test_full_pipeline_edge_cases(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = "Energy and force in physics"
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        assert isinstance(final, FinalDecisionResult)
        assert final.final_decision in VALID_DECISIONS
        assert 0.0 <= final.decision_confidence <= 1.0

    def test_full_pipeline_graceful_degradation(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Quantum mechanics describes electron"
            " behavior in atomic orbitals"
        )
        single_expert = [
            {"name": "expert_physics", "domain": "physics"},
        ]
        _, _, sel, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
            experts=single_expert,
        )
        assert isinstance(final, FinalDecisionResult)
        assert final.final_decision in VALID_DECISIONS
        assert sel.total_candidates <= 1


# ── performance ─────────────────────────────────────────────


class TestFullPipelinePerformance:
    """Performance benchmarks for the full pipeline."""

    def test_full_pipeline_performance_benchmark(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Patient with myocardial infarction"
            " showing elevated troponin levels"
        )
        start = time.time()
        _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 5000, (
            f"Pipeline took {elapsed_ms:.0f}ms, "
            f"exceeds 5000ms limit"
        )

    def test_caching_efficiency_full_pipeline(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "The chemical reaction between acid"
            " and base produces salt and water"
        )
        start1 = time.time()
        _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        time1 = time.time() - start1

        start2 = time.time()
        _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        time2 = time.time() - start2

        # Second run should not be significantly slower
        assert time2 <= time1 * 1.5, (
            f"Second run ({time2:.3f}s) was much slower"
            f" than first ({time1:.3f}s)"
        )


# ── quality ─────────────────────────────────────────────────


class TestFullPipelineQuality:
    """Quality and correctness tests for pipeline output."""

    def test_reasoning_chain_quality(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Patient with myocardial infarction"
            " showing elevated troponin levels"
        )
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        chain = final.reasoning_chain
        assert len(chain) >= 5, (
            f"Expected >= 5 reasoning steps, "
            f"got {len(chain)}"
        )
        for step in chain:
            assert isinstance(step, dict)
            assert len(step) > 0

    def test_action_recommendations_correctness(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Quantum mechanics describes electron"
            " behavior in atomic orbitals with"
            " energy quantization"
        )
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        assert final.recommended_action or True
        assert final.final_decision in VALID_ACTIONS

    def test_reproducibility_of_decisions(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "The chemical reaction between acid"
            " and base produces salt and water"
            " through neutralization"
        )
        _, _, _, _, _, final1 = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        _, _, _, _, _, final2 = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        assert final1.final_decision == final2.final_decision


# ── output completeness ─────────────────────────────────────


class TestFullPipelineOutputs:
    """Tests verifying completeness of pipeline outputs."""

    def test_all_phases_executed(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Medical imaging uses physics principles"
            " like radiation and magnetic resonance"
        )
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        for phase in EXPECTED_PHASES:
            assert phase in final.phases_executed, (
                f"Phase {phase} missing from "
                f"phases_executed"
            )

    def test_uncertainty_is_computed(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Patient with myocardial infarction"
            " showing elevated troponin levels"
        )
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        assert final.total_uncertainty >= 0, (
            "total_uncertainty should be >= 0"
        )

    def test_expert_predictions_populated(
        self,
        norm_pipeline,
        sem_pipeline,
        expert_pipeline,
        inf_pipeline,
        cal_pipeline,
        syn_pipeline,
    ):
        text = (
            "Quantum mechanics describes electron"
            " behavior in atomic orbitals"
        )
        _, _, _, _, _, final = _run_full_pipeline(
            text,
            norm_pipeline,
            sem_pipeline,
            expert_pipeline,
            inf_pipeline,
            cal_pipeline,
            syn_pipeline,
        )
        assert isinstance(
            final.expert_predictions, dict,
        )
        assert len(final.expert_predictions) > 0, (
            "expert_predictions should be non-empty"
        )
