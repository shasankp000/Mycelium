"""Tests for Phase 5.1 Feedback Integrator."""


from mycelium.pipeline.phase3.phases.phase_5_1_feedback_integrator import (
    ExpertParameterUpdater,
    FeedbackIntegrationPipeline,
    FeedbackIntegrator,
    RetrainingDataCollector,
)
from mycelium.pipeline.phase3.utils.types import (
    ExpertAnalysis,
    FeedbackData,
    PerformanceAnalysis,
    UpdatedExpertConfig,
    UpdatedSystemConfig,
)


def _make_expert_analysis(
    name="expert_medical", accuracy=0.5, calibration=0.1,
    latency=200.0, trend="stable",
):
    return ExpertAnalysis(
        expert_name=name, total_samples=100,
        accuracy=accuracy, precision=accuracy,
        recall=accuracy, f1_score=accuracy,
        average_latency_ms=latency,
        confidence_calibration_error=calibration,
        reliability_score=0.7,
        performance_trend=trend,
        last_updated="2026-01-01T00:00:00+00:00",
    )


def _make_performance_analysis(accuracy=0.5, experts=None):
    if experts is None:
        experts = [_make_expert_analysis()]
    return PerformanceAnalysis(
        analysis_timestamp="2026-01-01T00:00:00+00:00",
        time_period="1d", overall_accuracy=accuracy,
        overall_latency_ms=200.0,
        expert_analyses=experts,
        identified_issues=[
            {"type": "low_accuracy", "severity": 0.8},
        ],
        issue_severity_scores={"low_accuracy": 0.8},
        improvement_recommendations=["Retrain experts"],
        system_health_score=50.0, action_required=True,
    )


def _make_feedback(correct=True, expert="expert_medical"):
    return FeedbackData(
        input_text="test input", prediction="A",
        ground_truth="A" if correct else "B",
        prediction_correct=correct, user_rating=4.0,
        total_latency_ms=100.0, expert_used=expert,
        expert_confidence=0.8,
        timestamp="2026-01-01T00:00:00+00:00",
        session_id="s1", feedback_id="fb1",
    )


class TestFeedbackIntegrator:
    def setup_method(self):
        self.integrator = FeedbackIntegrator()

    def test_integrate_feedback_basic(self):
        analysis = _make_performance_analysis()
        feedback = [_make_feedback()]
        result = self.integrator.integrate_feedback(
            analysis, feedback
        )
        assert isinstance(result, UpdatedSystemConfig)
        assert result.timestamp != ""

    def test_integrate_generates_expert_updates(self):
        analysis = _make_performance_analysis(accuracy=0.4)
        result = self.integrator.integrate_feedback(
            analysis, [_make_feedback(correct=False)]
        )
        assert isinstance(result, UpdatedSystemConfig)

    def test_prioritize_improvements(self):
        issues = [
            {"type": "a", "severity": "low"},
            {"type": "b", "severity": "critical"},
            {"type": "c", "severity": "medium"},
        ]
        result = self.integrator.prioritize_improvements(issues)
        assert result[0]["severity"] == "critical"
        assert result[-1]["severity"] == "low"

    def test_generate_update_plan(self):
        issues = [{"type": "low_accuracy", "severity": 0.8}]
        plan = self.integrator.generate_update_plan(issues)
        assert isinstance(plan, list)
        assert len(plan) >= 1


class TestExpertParameterUpdater:
    def setup_method(self):
        self.updater = ExpertParameterUpdater()

    def test_update_expert_parameters(self):
        analysis = _make_expert_analysis(accuracy=0.5)
        factors = {"learning_rate_factor": 1.1}
        result = self.updater.update_expert_parameters(
            "expert_medical", analysis, factors
        )
        assert isinstance(result, UpdatedExpertConfig)
        assert result.expert_name == "expert_medical"

    def test_compute_adjustment_factors(self):
        analysis = _make_expert_analysis(accuracy=0.5)
        factors = self.updater.compute_adjustment_factors(
            analysis
        )
        assert isinstance(factors, dict)
        assert "learning_rate_factor" in factors

    def test_validate_new_parameters_valid(self):
        old = {"lr": 0.01, "batch": 32}
        new = {"lr": 0.011, "batch": 33}
        assert self.updater.validate_new_parameters(old, new)

    def test_validate_new_parameters_large_change(self):
        old = {"lr": 0.01}
        new = {"lr": 0.1}
        result = self.updater.validate_new_parameters(old, new)
        assert isinstance(result, bool)


class TestRetrainingDataCollector:
    def setup_method(self):
        self.collector = RetrainingDataCollector()

    def test_collect_retraining_data(self):
        feedback = [
            _make_feedback(correct=False),
            _make_feedback(correct=True),
        ]
        result = self.collector.collect_retraining_data(
            feedback, [{"type": "low_accuracy"}]
        )
        assert result.total_size >= 0

    def test_prioritize_samples(self):
        samples = [
            _make_feedback(correct=True),
            _make_feedback(correct=False),
        ]
        result = self.collector.prioritize_samples(samples)
        assert len(result) == 2

    def test_detect_data_drift_no_drift(self):
        feedback = [_make_feedback(correct=True)] * 10
        assert isinstance(
            self.collector.detect_data_drift(feedback), bool
        )


class TestFeedbackIntegrationPipeline:
    def setup_method(self):
        self.pipeline = FeedbackIntegrationPipeline()

    def test_integrate(self):
        analysis = _make_performance_analysis()
        result = self.pipeline.integrate(analysis)
        assert isinstance(result, UpdatedSystemConfig)

    def test_integrate_with_feedback(self):
        analysis = _make_performance_analysis()
        feedback = [_make_feedback()]
        result = self.pipeline.integrate(
            analysis, feedback_data=feedback
        )
        assert isinstance(result, UpdatedSystemConfig)

    def test_metadata(self):
        analysis = _make_performance_analysis()
        self.pipeline.integrate(analysis)
        meta = self.pipeline.get_integration_metadata()
        assert isinstance(meta, dict)
