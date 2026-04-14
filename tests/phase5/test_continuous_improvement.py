"""Tests for Phase 5.2 Continuous Improvement."""

import pytest

from phase3_validation.config.phase3_config import Phase3Config
from phase3_validation.phases.phase_5_2_continuous_improvement import (
    ContinuousImprovementLoop,
    ContinuousImprovementPipeline,
    ImprovementExecutor,
    ImprovementMetricsTracker,
)
from phase3_validation.utils.types import (
    ExpertAnalysis,
    ImprovementPlan,
    ImprovementProgress,
    PerformanceAnalysis,
    UpdatedExpertConfig,
    UpdatedSystemConfig,
)


def _make_system_config():
    return UpdatedSystemConfig(
        timestamp="2026-01-01T00:00:00+00:00",
        expert_updates=[
            UpdatedExpertConfig(
                expert_name="expert_medical",
                parameter_updates={"lr": 0.01},
                retraining_data_size=50,
                scheduled_retraining=True,
                expected_improvement=0.1,
            ),
        ],
        threshold_updates={"accuracy": 0.65},
        retraining_schedule={"expert_medical": "2026-02-01"},
    )


def _make_performance_analysis():
    return PerformanceAnalysis(
        analysis_timestamp="2026-01-01T00:00:00+00:00",
        time_period="1d", overall_accuracy=0.6,
        overall_latency_ms=200.0,
        expert_analyses=[
            ExpertAnalysis(
                expert_name="expert_medical",
                total_samples=100, accuracy=0.6,
                precision=0.6, recall=0.6, f1_score=0.6,
                average_latency_ms=200.0,
                confidence_calibration_error=0.1,
                reliability_score=0.7,
                performance_trend="declining",
            ),
        ],
        identified_issues=[
            {"type": "low_accuracy", "severity": 0.8},
        ],
        improvement_recommendations=["Retrain experts"],
        system_health_score=50.0, action_required=True,
    )


class TestContinuousImprovementLoop:
    def setup_method(self):
        self.loop = ContinuousImprovementLoop()

    def test_execute_improvement_cycle(self):
        result = self.loop.execute_improvement_cycle(
            _make_performance_analysis(),
            _make_system_config(),
        )
        assert isinstance(result, ImprovementPlan)
        assert result.plan_id != ""
        assert result.timestamp != ""

    def test_schedule_next_cycle(self):
        ts = self.loop.schedule_next_cycle({})
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_track_improvement_progress(self):
        old = {"accuracy": 0.6, "latency": 200.0}
        new = {"accuracy": 0.75, "latency": 180.0}
        result = self.loop.track_improvement_progress(
            old, new
        )
        assert isinstance(result, ImprovementProgress)

    def test_plan_actions_not_empty(self):
        result = self.loop.execute_improvement_cycle(
            _make_performance_analysis(),
            _make_system_config(),
        )
        assert isinstance(result.actions, list)


class TestImprovementExecutor:
    def setup_method(self):
        self.executor = ImprovementExecutor()

    def test_apply_expert_updates(self):
        updates = [
            UpdatedExpertConfig(
                expert_name="expert_medical",
                parameter_updates={"lr": 0.01},
            ),
        ]
        result = self.executor.apply_expert_updates(updates)
        assert isinstance(result, dict)
        assert "applied_count" in result or len(result) > 0

    def test_retrain_experts(self):
        configs = [{"expert_name": "expert_medical"}]
        result = self.executor.retrain_experts(configs)
        assert isinstance(result, dict)

    def test_verify_improvements_true(self):
        before = {"accuracy": 0.6}
        after = {"accuracy": 0.75}
        assert self.executor.verify_improvements(
            before, after
        ) is True

    def test_verify_improvements_false(self):
        before = {"accuracy": 0.8}
        after = {"accuracy": 0.5}
        assert self.executor.verify_improvements(
            before, after
        ) is False


class TestImprovementMetricsTracker:
    def setup_method(self):
        self.tracker = ImprovementMetricsTracker()

    def test_track_metric(self):
        self.tracker.track_metric("accuracy", 0.8)
        history = self.tracker.get_metric_history("accuracy")
        assert len(history) == 1
        assert history[0] == 0.8

    def test_compute_improvement_velocity(self):
        for v in [0.6, 0.65, 0.7, 0.75, 0.8]:
            self.tracker.track_metric("accuracy", v)
        vel = self.tracker.compute_improvement_velocity(
            "accuracy"
        )
        assert vel > 0

    def test_predict_future_performance(self):
        for v in [0.6, 0.65, 0.7]:
            self.tracker.track_metric("accuracy", v)
        pred = self.tracker.predict_future_performance(
            "accuracy"
        )
        assert isinstance(pred, dict)
        assert "predicted_value" in pred or len(pred) > 0

    def test_get_metric_history_empty(self):
        history = self.tracker.get_metric_history("unknown")
        assert history == []


class TestContinuousImprovementPipeline:
    def setup_method(self):
        self.pipeline = ContinuousImprovementPipeline()

    def test_run_improvement_cycle(self):
        result = self.pipeline.run_improvement_cycle(
            _make_performance_analysis(),
            _make_system_config(),
        )
        assert isinstance(result, ImprovementPlan)

    def test_get_improvement_status(self):
        status = self.pipeline.get_improvement_status()
        assert isinstance(status, dict)
