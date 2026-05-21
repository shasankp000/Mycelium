"""Tests for Phase 2.6 — Decision Synthesis."""

from types import SimpleNamespace

from mycelium.pipeline.phase2.phases.phase_2_6_synthesis import (
    ReasoningChainBuilder,
    DecisionMaker,
    ConfidenceScorer,
    ActionRecommender,
    DecisionSynthesisPipeline,
    FinalDecisionResult,
    DEFAULT_THRESHOLDS,
)
from mycelium.pipeline.phase2.phases.phase_2_4_inference import (
    InferenceResult,
)
from mycelium.pipeline.phase2.phases.phase_2_5_calibration import (
    CalibrationResult,
)
from mycelium.pipeline.phase2.utils.semantic_types import (
    ExpertSelectionResult,
    RankedExpert,
    SemanticResult,
    Concept,
)


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------


def _make_ranked(name, domain, score):
    return RankedExpert(
        name=name,
        domain=domain,
        match_score=score,
        confidence_low=max(score - 0.1, 0.0),
        confidence_high=min(score + 0.1, 1.0),
        ranking_factors={"domain": score},
    )


def _make_inference_result(n_experts=3):
    preds = {}
    for i in range(n_experts):
        name = f"expert_{i}"
        preds[name] = {
            "prediction": {
                "recommended_action": "use_existing",
                "domain": f"d{i}",
            },
            "confidence": 0.5 + i * 0.1,
            "metadata": {"expert_name": name},
            "latency_ms": 1.0,
        }
    return InferenceResult(
        text="test",
        selected_experts=[
            f"expert_{i}" for i in range(n_experts)
        ],
        predictions_per_expert=preds,
        ensemble_confidence=0.7,
        prediction_disagreement=0.1,
        prediction_confidences={
            f"expert_{i}": 0.5 + i * 0.1
            for i in range(n_experts)
        },
    )


def _make_calibration_result(n_experts=3):
    calibrated = []
    for i in range(n_experts):
        calibrated.append({
            "prediction": {
                "recommended_action": "use_existing",
                "domain": f"d{i}",
            },
            "confidence": 0.6 + i * 0.05,
            "calibrated_confidence": 0.55 + i * 0.05,
            "expert_name": f"expert_{i}",
        })
    return CalibrationResult(
        calibrated_predictions=calibrated,
        calibration_method="temperature_scaling",
        temperature=1.1,
        epistemic_uncertainty={
            "variance": 0.02,
            "entropy": 0.3,
        },
        aleatoric_uncertainty={"data_entropy": 0.2},
        total_uncertainty={
            "total_uncertainty": 0.22,
            "epistemic_fraction": 0.5,
            "aleatoric_fraction": 0.5,
        },
        bayesian_posterior={
            "use_existing": 0.7,
            "create_patch": 0.2,
            "create_new_expert": 0.1,
        },
        uncertainty_intervals={
            "expert_0": (0.5, 0.7),
        },
        calibration_metrics={"mean_confidence": 0.65},
    )


def _make_selection_result(n_experts=3):
    experts = [
        _make_ranked(
            f"expert_{i}", f"d{i}", 0.7 + i * 0.05
        )
        for i in range(n_experts)
    ]
    return ExpertSelectionResult(
        selected_experts=experts,
        total_candidates=5,
        total_selected=n_experts,
        expert_scores={
            e.name: e.match_score for e in experts
        },
        confidence_in_selection=0.75,
        selection_strategy="greedy",
    )


def _make_semantic_result():
    return SemanticResult(
        original_text="test",
        cleaned_text="test",
        domain_relevance_scores={
            "medical": 0.8,
            "physics": 0.3,
        },
        ranked_domains=["medical", "physics"],
        confidence_score=0.7,
        key_concepts=[
            Concept(text="treatment"),
            Concept(text="therapy"),
        ],
    )


def _make_norm_result():
    return SimpleNamespace(
        cleaned_text="test cleaned text",
        original_text="test original text",
        language="en",
        language_confidence=0.95,
        is_valid=True,
        extracted_tags=["medical"],
        tag_confidence_scores={"medical": 0.9},
        domain_mapping={"medical": "medical"},
        processing_time_ms=1.0,
    )


# ==============================================================
# TestReasoningChainBuilder
# ==============================================================


class TestReasoningChainBuilder:
    """Tests for ReasoningChainBuilder."""

    def setup_method(self):
        self.builder = ReasoningChainBuilder()
        self.norm = _make_norm_result()
        self.sem = _make_semantic_result()
        self.sel = _make_selection_result()
        self.inf = _make_inference_result()
        self.cal = _make_calibration_result()

    def test_reasoning_chain_construction(self):
        chain = self.builder.build_reasoning_chain(
            self.norm, self.sem, self.sel,
            self.inf, self.cal,
        )
        assert len(chain) == 5
        required = {
            "step", "phase", "name",
            "description", "details",
        }
        for step in chain:
            assert required.issubset(step.keys())

    def test_reasoning_text_generation(self):
        chain = self.builder.build_reasoning_chain(
            self.norm, self.sem, self.sel,
            self.inf, self.cal,
        )
        text = self.builder.generate_reasoning_text(chain)
        assert "Reasoning Chain:" in text
        for step in chain:
            assert step["name"] in text

    def test_evidence_addition(self):
        chain = self.builder.build_reasoning_chain(
            self.norm, self.sem, self.sel,
            self.inf, self.cal,
        )
        evidence = {
            "2.2": {"extra_info": "domain match"},
        }
        updated = self.builder.add_evidence(
            chain, evidence,
        )
        phase_22 = [
            s for s in updated if s["phase"] == "2.2"
        ]
        assert len(phase_22) == 1
        assert "evidence" in phase_22[0]
        assert (
            phase_22[0]["evidence"]["extra_info"]
            == "domain match"
        )


# ==============================================================
# TestDecisionMaker
# ==============================================================


class TestDecisionMaker:
    """Tests for DecisionMaker."""

    def setup_method(self):
        self.maker = DecisionMaker()
        self.cal = _make_calibration_result()
        self.sel = _make_selection_result()

    def test_decision_making(self):
        result = self.maker.make_decision(
            self.cal, self.sel,
        )
        assert "decision" in result

    def test_decision_rule_application(self):
        preds = self.cal.calibrated_predictions
        result = self.maker.apply_decision_rules(
            preds, DEFAULT_THRESHOLDS,
        )
        assert "decision" in result
        assert "confidence" in result

    def test_boundary_case_detection(self):
        preds = [
            {"confidence": 0.61},
            {"confidence": 0.59},
        ]
        is_boundary = (
            self.maker.detect_decision_boundary_cases(
                preds, DEFAULT_THRESHOLDS,
            )
        )
        assert is_boundary is True


# ==============================================================
# TestConfidenceScorer
# ==============================================================


class TestConfidenceScorer:
    """Tests for ConfidenceScorer."""

    def setup_method(self):
        self.scorer = ConfidenceScorer()
        self.cal = _make_calibration_result()
        self.inf = _make_inference_result()

    def test_final_confidence_scoring(self):
        score = self.scorer.score_final_confidence(
            self.cal, self.inf, expert_count=3,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_confidence_factor_breakdown(self):
        results = {
            "semantic": _make_semantic_result(),
            "selection": _make_selection_result(),
            "inference": self.inf,
            "calibration": self.cal,
        }
        factors = (
            self.scorer.compute_confidence_factors(results)
        )
        assert "semantic_confidence" in factors
        assert "selection_confidence" in factors
        assert "ensemble_confidence" in factors
        assert "mean_calibrated_confidence" in factors

    def test_confidence_penalty_application(self):
        base = 0.9
        penalized = self.scorer.apply_confidence_penalties(
            base, uncertainty=0.3, disagreement=0.2,
        )
        assert penalized < base


# ==============================================================
# TestActionRecommender
# ==============================================================


class TestActionRecommender:
    """Tests for ActionRecommender."""

    def setup_method(self):
        self.recommender = ActionRecommender()
        self.sem = _make_semantic_result()
        self.inf = _make_inference_result()
        self.sel = _make_selection_result()

    def test_action_recommendation(self):
        decision = {
            "decision": "use_existing",
            "confidence": 0.8,
        }
        domain_analysis = {"medical": 0.8}
        result = self.recommender.recommend_action(
            decision, self.sel, domain_analysis,
        )
        assert "action" in result

    def test_new_expert_recommendation(self):
        result = (
            self.recommender.recommend_new_expert_creation(
                self.sem, self.inf,
            )
        )
        assert result["action"] == "create_new_expert"
        assert "suggested_domain" in result
        assert "key_concepts" in result
        assert "confidence" in result
        assert "priority" in result

    def test_patch_recommendation(self):
        result = (
            self.recommender.recommend_patch_creation(
                self.sem, self.inf,
            )
        )
        assert result["action"] == "create_patch"
        assert "target_expert" in result
        assert "gap_concepts" in result
        assert "confidence" in result


# ==============================================================
# TestDecisionSynthesisPipeline
# ==============================================================


class TestDecisionSynthesisPipeline:
    """Tests for DecisionSynthesisPipeline."""

    def setup_method(self):
        self.pipeline = DecisionSynthesisPipeline()
        self.norm = _make_norm_result()
        self.sem = _make_semantic_result()
        self.sel = _make_selection_result()
        self.inf = _make_inference_result()
        self.cal = _make_calibration_result()

    def test_full_synthesis_pipeline(self):
        result = self.pipeline.synthesize(
            self.norm, self.sem, self.sel,
            self.inf, self.cal,
        )
        assert isinstance(result, FinalDecisionResult)
        assert result.original_text != ""
        assert result.final_decision in {
            "use_existing",
            "create_patch",
            "create_new_expert",
        }
        assert 0.0 <= result.decision_confidence <= 1.0
        assert len(result.reasoning_chain) == 5
        assert result.reasoning_text != ""
        assert result.recommended_action != ""
        assert len(result.selected_experts) > 0
        assert result.primary_expert is not None
        assert result.processing_time_ms >= 0.0

    def test_synthesis_with_custom_thresholds(self):
        strict = {
            "use_existing": 0.99,
            "create_patch": 0.98,
            "create_new_expert": 0.0,
        }
        result = self.pipeline.synthesize(
            self.norm, self.sem, self.sel,
            self.inf, self.cal,
            decision_thresholds=strict,
        )
        assert result.final_decision in {
            "create_patch",
            "create_new_expert",
        }

    def test_synthesis_phases_executed(self):
        result = self.pipeline.synthesize(
            self.norm, self.sem, self.sel,
            self.inf, self.cal,
        )
        expected = [
            "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
        ]
        assert result.phases_executed == expected

    def test_processing_metadata(self):
        self.pipeline.synthesize(
            self.norm, self.sem, self.sel,
            self.inf, self.cal,
        )
        meta = self.pipeline.get_processing_metadata()
        assert isinstance(meta, dict)
        assert "decision" in meta
        assert "confidence" in meta
        assert "processing_time_ms" in meta

    def test_metadata_empty_before_synthesis(self):
        fresh = DecisionSynthesisPipeline()
        meta = fresh.get_processing_metadata()
        assert meta == {}
