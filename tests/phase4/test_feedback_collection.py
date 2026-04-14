"""Tests for Phase 4.1 – Feedback Collection.

Covers FeedbackCollector, PerformanceMetricsCompute,
FeedbackAggregator, and FeedbackCollectionPipeline.
"""

import pytest

from phase3_validation.config.phase3_config import Phase3Config
from phase3_validation.phases.phase_4_1_feedback_collector import (
    FeedbackCollector,
    FeedbackCollectionPipeline,
    FeedbackAggregator,
    PerformanceMetricsCompute,
    FeedbackCollectionError,
)
from phase3_validation.utils.types import (
    AggregatedFeedback,
    FeedbackData,
)


# ── helpers ──────────────────────────────────────────────────────


def _make_feedback(
    *,
    input_text: str = "q",
    prediction: object = "a",
    ground_truth: object = None,
    prediction_correct: bool | None = None,
    user_rating: float | None = None,
    total_latency_ms: float = 10.0,
    expert_used: str = "expert_a",
    expert_confidence: float = 0.9,
) -> FeedbackData:
    return FeedbackData(
        input_text=input_text,
        prediction=prediction,
        ground_truth=ground_truth,
        prediction_correct=prediction_correct,
        user_rating=user_rating,
        total_latency_ms=total_latency_ms,
        expert_used=expert_used,
        expert_confidence=expert_confidence,
        timestamp="2025-01-01T00:00:00+00:00",
        session_id="session_test",
        feedback_id="fb_test",
    )


# ── FeedbackCollector ────────────────────────────────────────────


class TestFeedbackCollector:
    def setup_method(self):
        self.config = Phase3Config()
        self.collector = FeedbackCollector(config=self.config)

    def test_collect_feedback_basic(self):
        fb = self.collector.collect_feedback(
            input_text="hello", prediction="world",
        )
        assert isinstance(fb, FeedbackData)
        assert fb.input_text == "hello"
        assert fb.prediction == "world"
        assert fb.ground_truth is None
        assert fb.prediction_correct is None
        assert fb.feedback_id.startswith("fb_")
        assert fb.session_id.startswith("session_")

    def test_collect_with_ground_truth(self):
        fb = self.collector.collect_feedback(
            input_text="q", prediction="correct", ground_truth="correct",
        )
        assert fb.prediction_correct is True

        fb2 = self.collector.collect_feedback(
            input_text="q", prediction="wrong", ground_truth="correct",
        )
        assert fb2.prediction_correct is False

    def test_collect_with_user_rating(self):
        fb = self.collector.collect_feedback(
            input_text="q", prediction="a", user_rating=4.5,
        )
        assert fb.user_rating == 4.5

    def test_collect_with_execution_metrics(self):
        metrics = {
            "total_latency_ms": 123.4,
            "phase_latencies": {"phase1": 50.0, "phase2": 73.4},
            "expert_used": "bio_expert",
            "expert_confidence": 0.85,
        }
        fb = self.collector.collect_feedback(
            input_text="q", prediction="a", execution_metrics=metrics,
        )
        assert fb.total_latency_ms == 123.4
        assert fb.phase_latencies == {"phase1": 50.0, "phase2": 73.4}
        assert fb.expert_used == "bio_expert"
        assert fb.expert_confidence == 0.85

    def test_record_prediction_latency(self):
        self.collector.record_prediction_latency("routing", 15.0)
        self.collector.record_prediction_latency("routing", 20.0)
        assert self.collector._latency_records["routing"] == [15.0, 20.0]

    def test_record_user_feedback(self):
        self.collector.record_user_feedback(4.0, "great")
        assert len(self.collector._user_feedback_log) == 1
        entry = self.collector._user_feedback_log[0]
        assert entry["rating"] == 4.0
        assert entry["comments"] == "great"
        assert "timestamp" in entry

    def test_record_expert_contribution(self):
        self.collector.record_expert_contribution("chem_expert", 0.92)
        self.collector.record_expert_contribution("chem_expert", 0.88)
        assert self.collector._expert_contributions["chem_expert"] == [
            0.92, 0.88,
        ]

    def test_feedback_stored_in_history(self):
        self.collector.collect_feedback(input_text="a", prediction="b")
        self.collector.collect_feedback(input_text="c", prediction="d")
        assert len(self.collector._feedback_history) == 2


# ── PerformanceMetricsCompute ────────────────────────────────────


class TestPerformanceMetricsCompute:
    def setup_method(self):
        self.pmc = PerformanceMetricsCompute()

    def test_compute_accuracy(self):
        preds = ["a", "b", "c", "d"]
        truth = ["a", "x", "c", "d"]
        assert self.pmc.compute_accuracy(preds, truth) == pytest.approx(0.75)

    def test_compute_accuracy_empty(self):
        assert self.pmc.compute_accuracy([], []) == 0.0
        assert self.pmc.compute_accuracy(["a"], []) == 0.0

    def test_compute_calibration_error(self):
        confidences = [0.95, 0.85, 0.15, 0.05]
        accuracies = [True, True, False, False]
        ece = self.pmc.compute_calibration_error(confidences, accuracies)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_compute_calibration_error_empty(self):
        assert self.pmc.compute_calibration_error([], []) == 0.0

    def test_compute_latency_metrics(self):
        latencies = {"phase1": 10.0, "phase2": 30.0, "phase3": 20.0}
        result = self.pmc.compute_latency_metrics(latencies)
        assert result["mean"] == pytest.approx(20.0)
        assert result["median"] == pytest.approx(20.0)
        assert result["min"] == pytest.approx(10.0)
        assert result["max"] == pytest.approx(30.0)
        assert "p95" in result

    def test_compute_latency_metrics_empty(self):
        result = self.pmc.compute_latency_metrics({})
        assert result["mean"] == 0.0
        assert result["p95"] == 0.0

    def test_compute_expert_performance(self):
        fb_list = [
            _make_feedback(
                expert_used="bio", prediction="a", ground_truth="a",
                prediction_correct=True, total_latency_ms=10.0,
                expert_confidence=0.9,
            ),
            _make_feedback(
                expert_used="bio", prediction="b", ground_truth="c",
                prediction_correct=False, total_latency_ms=20.0,
                expert_confidence=0.8,
            ),
            _make_feedback(expert_used="chem"),
        ]
        perf = self.pmc.compute_expert_performance("bio", fb_list)
        assert perf["accuracy"] == pytest.approx(0.5)
        assert perf["average_latency_ms"] == pytest.approx(15.0)
        assert perf["average_confidence"] == pytest.approx(0.85)

    def test_compute_expert_performance_no_match(self):
        perf = self.pmc.compute_expert_performance("missing", [])
        assert perf["accuracy"] == 0.0
        assert perf["average_latency_ms"] == 0.0


# ── FeedbackAggregator ──────────────────────────────────────────


class TestFeedbackAggregator:
    def setup_method(self):
        self.aggregator = FeedbackAggregator()

    def test_aggregate_feedback_basic(self):
        fb_list = [
            _make_feedback(
                prediction_correct=True, total_latency_ms=10.0,
                user_rating=5.0, expert_used="e1",
            ),
            _make_feedback(
                prediction_correct=False, total_latency_ms=20.0,
                user_rating=3.0, expert_used="e1",
            ),
        ]
        agg = self.aggregator.aggregate_feedback(fb_list)
        assert isinstance(agg, AggregatedFeedback)
        assert agg.num_samples == 2
        assert agg.overall_accuracy == pytest.approx(0.5)
        assert agg.average_latency_ms == pytest.approx(15.0)
        assert agg.user_satisfaction == pytest.approx(4.0)
        assert agg.time_period == "all"

    def test_aggregate_feedback_custom_window(self):
        agg = self.aggregator.aggregate_feedback([], time_window="1h")
        assert agg.time_period == "1h"
        assert agg.num_samples == 0

    def test_compute_trend_improving(self):
        values = [0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9]
        assert self.aggregator.compute_trend(values) == "improving"

    def test_compute_trend_declining(self):
        values = [0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5]
        assert self.aggregator.compute_trend(values) == "declining"

    def test_compute_trend_stable(self):
        values = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        assert self.aggregator.compute_trend(values) == "stable"

    def test_compute_trend_short_list(self):
        assert self.aggregator.compute_trend([1.0, 2.0]) == "stable"

    def test_identify_failure_modes(self):
        fb_list = [
            _make_feedback(prediction_correct=True, expert_used="e1"),
            _make_feedback(prediction_correct=False, expert_used="e1"),
            _make_feedback(prediction_correct=False, expert_used="e2"),
            _make_feedback(prediction_correct=False, expert_used="e2"),
        ]
        result = self.aggregator.identify_failure_modes(fb_list)
        assert result["total_failures"] == 3
        assert result["failure_rate"] == pytest.approx(0.75)
        assert result["expert_failure_counts"]["e1"] == 1
        assert result["expert_failure_counts"]["e2"] == 2

    def test_identify_failure_modes_none(self):
        fb_list = [
            _make_feedback(prediction_correct=True, expert_used="e1"),
        ]
        result = self.aggregator.identify_failure_modes(fb_list)
        assert result["total_failures"] == 0
        assert result["failure_rate"] == pytest.approx(0.0)


# ── FeedbackCollectionPipeline ──────────────────────────────────


class TestFeedbackCollectionPipeline:
    def setup_method(self):
        self.pipeline = FeedbackCollectionPipeline()

    def test_pipeline_collect(self):
        fb = self.pipeline.collect({
            "input_text": "question",
            "prediction": "answer",
            "ground_truth": "answer",
        })
        assert isinstance(fb, FeedbackData)
        assert fb.prediction_correct is True
        assert fb.input_text == "question"

    def test_pipeline_aggregate(self):
        fb1 = self.pipeline.collect({
            "input_text": "q1", "prediction": "a",
            "ground_truth": "a",
            "execution_metrics": {"total_latency_ms": 5.0},
        })
        fb2 = self.pipeline.collect({
            "input_text": "q2", "prediction": "b",
            "ground_truth": "c",
            "execution_metrics": {"total_latency_ms": 15.0},
        })
        agg = self.pipeline.aggregate([fb1, fb2])
        assert isinstance(agg, AggregatedFeedback)
        assert agg.num_samples == 2
        assert agg.overall_accuracy == pytest.approx(0.5)
        assert agg.average_latency_ms == pytest.approx(10.0)

    def test_pipeline_metadata(self):
        self.pipeline.collect({
            "input_text": "q", "prediction": "a",
        })
        meta = self.pipeline.get_collection_metadata()
        assert "feedback_id" in meta
        assert "session_id" in meta
        assert "collection_time_ms" in meta
        assert "timestamp" in meta
        assert meta["collection_time_ms"] >= 0.0

    def test_pipeline_metadata_empty_before_collect(self):
        meta = self.pipeline.get_collection_metadata()
        assert meta == {}

    def test_pipeline_uses_custom_config(self):
        config = Phase3Config()
        pipeline = FeedbackCollectionPipeline(config=config)
        assert pipeline._config is config
