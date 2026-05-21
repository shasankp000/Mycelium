"""Tests for Phase 4.2 — Performance Analyzer components."""

import pytest

from mycelium.pipeline.phase3.config.phase3_config import Phase3Config
from mycelium.pipeline.phase3.phases.phase_4_2_performance_analyzer import (
    PerformanceAnalyzer,
    ExpertPerformanceAnalyzer,
    PerformanceAnalysisPipeline,
    PerformanceAnalysisError,
)
from mycelium.pipeline.phase3.utils.types import (
    AggregatedFeedback,
    ExpertAnalysis,
    PerformanceAnalysis,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_feedback(**kwargs) -> AggregatedFeedback:
    """Build an AggregatedFeedback with sensible defaults."""
    defaults = {
        "time_period": "1d",
        "num_samples": 100,
        "overall_accuracy": 0.85,
        "overall_calibration_error": 0.05,
        "average_latency_ms": 150.0,
        "expert_performance": {
            "expert_medical": {
                "accuracy": 0.9,
                "latency_ms": 120.0,
                "confidence": 0.88,
                "sample_count": 50,
                "calibration_error": 0.04,
                "average_latency_ms": 120.0,
                "confidence_calibration_error": 0.04,
                "total_samples": 50,
                "precision": 0.88,
                "recall": 0.85,
                "performance_trend": "stable",
            },
        },
        "user_satisfaction": 4.2,
        "trends": {"accuracy": "stable", "latency": "stable"},
    }
    defaults.update(kwargs)
    return AggregatedFeedback(**defaults)


def _make_unhealthy_feedback(**kwargs) -> AggregatedFeedback:
    """Build feedback with poor metrics across the board."""
    defaults = {
        "time_period": "1d",
        "num_samples": 200,
        "overall_accuracy": 0.40,
        "overall_calibration_error": 0.25,
        "average_latency_ms": 800.0,
        "expert_performance": {
            "expert_weak": {
                "accuracy": 0.35,
                "average_latency_ms": 700.0,
                "confidence_calibration_error": 0.20,
                "total_samples": 100,
                "precision": 0.30,
                "recall": 0.25,
                "performance_trend": "declining",
            },
        },
        "user_satisfaction": 1.5,
        "trends": {"accuracy": "declining", "latency": "declining"},
    }
    defaults.update(kwargs)
    return AggregatedFeedback(**defaults)


# ------------------------------------------------------------------
# TestPerformanceAnalyzer
# ------------------------------------------------------------------


class TestPerformanceAnalyzer:
    """Tests for the PerformanceAnalyzer class."""

    def setup_method(self):
        self.config = Phase3Config()
        self.analyzer = PerformanceAnalyzer(config=self.config)

    def test_analyze_performance_healthy(self):
        """Good metrics should yield a high health score."""
        feedback = _make_feedback(
            overall_accuracy=0.95,
            average_latency_ms=50.0,
            user_satisfaction=4.8,
        )
        result = self.analyzer.analyze_performance(feedback)

        assert isinstance(result, PerformanceAnalysis)
        assert result.system_health_score > 70.0
        assert result.action_required is False
        assert result.overall_accuracy == 0.95
        assert result.time_period == "1d"

    def test_analyze_performance_unhealthy(self):
        """Bad metrics should yield a low health score with action required."""
        feedback = _make_unhealthy_feedback()
        result = self.analyzer.analyze_performance(feedback)

        assert isinstance(result, PerformanceAnalysis)
        assert result.system_health_score < 60.0
        assert result.action_required is True
        assert len(result.identified_issues) > 0
        assert len(result.improvement_recommendations) > 0

    def test_identify_low_accuracy_issue(self):
        """Accuracy below threshold should raise a low_accuracy issue."""
        feedback = _make_feedback(overall_accuracy=0.40)
        issues = self.analyzer.identify_performance_issues(feedback)

        low_acc = [i for i in issues if i["type"] == "low_accuracy"]
        assert len(low_acc) == 1
        assert low_acc[0]["severity"] == "high"
        assert low_acc[0]["metric_value"] == 0.40

    def test_identify_high_latency_issue(self):
        """Latency above threshold should raise a high_latency issue."""
        feedback = _make_feedback(average_latency_ms=6000.0)
        issues = self.analyzer.identify_performance_issues(feedback)

        high_lat = [i for i in issues if i["type"] == "high_latency"]
        assert len(high_lat) == 1
        assert high_lat[0]["metric_value"] == 6000.0

    def test_identify_low_user_satisfaction(self):
        """Satisfaction below 3.0 should raise a low_user_satisfaction issue."""
        feedback = _make_feedback(user_satisfaction=2.0)
        issues = self.analyzer.identify_performance_issues(feedback)

        low_sat = [i for i in issues if i["type"] == "low_user_satisfaction"]
        assert len(low_sat) == 1
        assert low_sat[0]["metric_value"] == 2.0
        assert low_sat[0]["threshold"] == 3.0

    def test_identify_high_calibration_error(self):
        """Calibration error above 0.15 should raise an issue."""
        feedback = _make_feedback(overall_calibration_error=0.30)
        issues = self.analyzer.identify_performance_issues(feedback)

        cal_issues = [i for i in issues if i["type"] == "high_calibration_error"]
        assert len(cal_issues) == 1
        assert cal_issues[0]["metric_value"] == 0.30

    def test_identify_declining_trend(self):
        """Declining trends should raise declining_trend issues."""
        feedback = _make_feedback(
            trends={"accuracy": "declining", "latency": "stable"},
        )
        issues = self.analyzer.identify_performance_issues(feedback)

        declining = [i for i in issues if i["type"] == "declining_trend"]
        assert len(declining) == 1
        assert declining[0]["severity"] == "low"

    def test_compute_issue_severity(self):
        """Severity score should be clamped to [0, 1]."""
        issue = {
            "type": "low_accuracy",
            "metric_value": 0.50,
            "threshold": 0.80,
        }
        severity = self.analyzer.compute_issue_severity(issue)

        assert 0.0 <= severity <= 1.0
        assert severity >= 0.8  # base severity for low_accuracy

    def test_compute_issue_severity_unknown_type(self):
        """Unknown issue types should default to base 0.5."""
        issue = {
            "type": "unknown_issue",
            "metric_value": 0.0,
            "threshold": 1.0,
        }
        severity = self.analyzer.compute_issue_severity(issue)
        assert 0.0 <= severity <= 1.0

    def test_generate_recommendations(self):
        """Each issue type should produce its own recommendation."""
        issues = [
            {"type": "low_accuracy"},
            {"type": "high_latency"},
            {"type": "low_user_satisfaction"},
        ]
        recs = self.analyzer.generate_improvement_recommendations(issues)

        assert len(recs) == 3
        assert any("accuracy" in r.lower() for r in recs)
        assert any("latency" in r.lower() for r in recs)
        assert any("user experience" in r.lower() or "satisfaction" in r.lower() for r in recs)

    def test_generate_recommendations_deduplicates(self):
        """Duplicate issue types should not duplicate recommendations."""
        issues = [
            {"type": "low_accuracy"},
            {"type": "low_accuracy"},
        ]
        recs = self.analyzer.generate_improvement_recommendations(issues)
        assert len(recs) == 1

    def test_no_issues_for_good_metrics(self):
        """Perfect metrics should produce zero issues."""
        feedback = _make_feedback(
            overall_accuracy=0.99,
            average_latency_ms=10.0,
            user_satisfaction=5.0,
            overall_calibration_error=0.01,
            trends={"accuracy": "improving"},
        )
        issues = self.analyzer.identify_performance_issues(feedback)
        assert len(issues) == 0


# ------------------------------------------------------------------
# TestExpertPerformanceAnalyzer
# ------------------------------------------------------------------


class TestExpertPerformanceAnalyzer:
    """Tests for the ExpertPerformanceAnalyzer class."""

    def setup_method(self):
        self.config = Phase3Config()
        self.analyzer = ExpertPerformanceAnalyzer(config=self.config)

    def test_analyze_expert_good(self):
        """Expert with good metrics should produce a solid analysis."""
        feedback = _make_feedback()
        result = self.analyzer.analyze_expert("expert_medical", feedback)

        assert isinstance(result, ExpertAnalysis)
        assert result.expert_name == "expert_medical"
        assert result.accuracy == 0.9
        assert result.total_samples == 50
        assert result.performance_trend == "stable"
        assert result.last_updated != ""

    def test_analyze_expert_poor(self):
        """Expert with poor metrics should reflect low scores."""
        feedback = _make_unhealthy_feedback()
        result = self.analyzer.analyze_expert("expert_weak", feedback)

        assert isinstance(result, ExpertAnalysis)
        assert result.accuracy == 0.35
        assert result.performance_trend == "declining"
        assert result.reliability_score < 0.6

    def test_analyze_expert_missing_raises(self):
        """Requesting an absent expert should raise an error."""
        feedback = _make_feedback()
        with pytest.raises(PerformanceAnalysisError, match="not found"):
            self.analyzer.analyze_expert("nonexistent_expert", feedback)

    def test_detect_degradation_true(self):
        """Expert with declining trend should be flagged."""
        feedback = _make_unhealthy_feedback()
        degraded = self.analyzer.detect_expert_degradation(
            "expert_weak", feedback,
        )
        assert degraded is True

    def test_detect_degradation_false(self):
        """Expert with stable good metrics should not be degraded."""
        feedback = _make_feedback()
        degraded = self.analyzer.detect_expert_degradation(
            "expert_medical", feedback,
        )
        assert degraded is False

    def test_reliability_score(self):
        """Reliability score should be clamped to [0, 1]."""
        feedback = _make_feedback()
        score = self.analyzer.compute_expert_reliability_score(
            "expert_medical", feedback,
        )
        assert 0.0 <= score <= 1.0

    def test_reliability_score_poor_expert(self):
        """Weak expert should have lower reliability."""
        feedback = _make_unhealthy_feedback()
        score = self.analyzer.compute_expert_reliability_score(
            "expert_weak", feedback,
        )
        assert 0.0 <= score <= 1.0
        assert score < 0.7

    def test_analyze_expert_computes_f1(self):
        """F1 should be computed from precision and recall."""
        feedback = _make_feedback()
        result = self.analyzer.analyze_expert("expert_medical", feedback)

        expected_f1 = 2 * 0.88 * 0.85 / (0.88 + 0.85)
        assert abs(result.f1_score - expected_f1) < 0.01


# ------------------------------------------------------------------
# TestPerformanceAnalysisPipeline
# ------------------------------------------------------------------


class TestPerformanceAnalysisPipeline:
    """Tests for the PerformanceAnalysisPipeline class."""

    def setup_method(self):
        self.config = Phase3Config()
        self.pipeline = PerformanceAnalysisPipeline(config=self.config)

    def test_pipeline_analysis(self):
        """Full pipeline should return a PerformanceAnalysis."""
        feedback = _make_feedback()
        result = self.pipeline.analyze(feedback)

        assert isinstance(result, PerformanceAnalysis)
        assert result.analysis_timestamp != ""
        assert result.time_period == "1d"
        assert result.overall_accuracy == 0.85

    def test_pipeline_with_experts(self):
        """Pipeline should analyze all experts in feedback."""
        feedback = _make_feedback(
            expert_performance={
                "expert_a": {
                    "accuracy": 0.90,
                    "average_latency_ms": 100.0,
                    "confidence_calibration_error": 0.03,
                    "total_samples": 40,
                    "precision": 0.88,
                    "recall": 0.86,
                    "performance_trend": "stable",
                },
                "expert_b": {
                    "accuracy": 0.80,
                    "average_latency_ms": 200.0,
                    "confidence_calibration_error": 0.08,
                    "total_samples": 60,
                    "precision": 0.75,
                    "recall": 0.78,
                    "performance_trend": "improving",
                },
            },
        )
        result = self.pipeline.analyze(feedback)

        assert len(result.expert_analyses) == 2
        names = {ea.expert_name for ea in result.expert_analyses}
        assert "expert_a" in names
        assert "expert_b" in names

    def test_pipeline_metadata(self):
        """Metadata should be populated after analysis."""
        feedback = _make_feedback()
        self.pipeline.analyze(feedback)
        metadata = self.pipeline.get_analysis_metadata()

        assert isinstance(metadata, dict)
        assert "timestamp" in metadata
        assert "elapsed_ms" in metadata
        assert "num_experts_analyzed" in metadata
        assert metadata["num_experts_analyzed"] == 1
        assert "health_score" in metadata

    def test_pipeline_action_required(self):
        """Unhealthy feedback should set action_required=True."""
        feedback = _make_unhealthy_feedback()
        result = self.pipeline.analyze(feedback)

        assert result.action_required is True
        metadata = self.pipeline.get_analysis_metadata()
        assert metadata["action_required"] is True

    def test_pipeline_no_action_required(self):
        """Healthy feedback should not require action."""
        feedback = _make_feedback(
            overall_accuracy=0.95,
            average_latency_ms=50.0,
            user_satisfaction=4.8,
        )
        result = self.pipeline.analyze(feedback)

        assert result.action_required is False

    def test_pipeline_metadata_empty_before_analysis(self):
        """Metadata should be empty before any analysis runs."""
        metadata = self.pipeline.get_analysis_metadata()
        assert metadata == {}
