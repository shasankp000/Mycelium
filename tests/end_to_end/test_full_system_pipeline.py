"""End-to-end tests for the full Phase 3-5 system pipeline."""

import pytest

from phase3_validation.config.phase3_config import Phase3Config
from phase3_validation.pipeline import Phase3To5Pipeline
from phase3_validation.utils.types import (
    ActionResult,
    FeedbackData,
    FinalDecisionResult,
    SystemExecutionResult,
)


def _make_decision(action="use_existing", domain="medical"):
    return FinalDecisionResult(
        decision="Use existing medical expert",
        confidence=0.85, reasoning="High domain match",
        action=action, expert_name="expert_medical",
        domain=domain,
    )


class TestFullSystemPipeline:
    def setup_method(self):
        self.pipeline = Phase3To5Pipeline()

    def test_complete_flow_use_existing(self):
        result = self.pipeline.run_complete_pipeline(
            _make_decision("use_existing")
        )
        assert isinstance(result, SystemExecutionResult)
        assert result.success is True
        assert result.action_result is not None

    def test_complete_flow_create_new(self):
        result = self.pipeline.run_complete_pipeline(
            _make_decision("create_new")
        )
        assert isinstance(result, SystemExecutionResult)
        assert result.success is True

    def test_complete_flow_create_patch(self):
        result = self.pipeline.run_complete_pipeline(
            _make_decision("create_patch")
        )
        assert isinstance(result, SystemExecutionResult)
        assert result.success is True

    def test_feedback_data_collected(self):
        result = self.pipeline.run_complete_pipeline(
            _make_decision()
        )
        assert result.feedback_data is not None
        assert isinstance(result.feedback_data, FeedbackData)

    def test_with_ground_truth(self):
        result = self.pipeline.run_complete_pipeline(
            _make_decision(), ground_truth="expected"
        )
        assert result.success is True

    def test_with_user_rating(self):
        result = self.pipeline.run_complete_pipeline(
            _make_decision(), user_rating=4.5
        )
        assert result.success is True

    def test_pipeline_metrics(self):
        self.pipeline.run_complete_pipeline(
            _make_decision()
        )
        metrics = self.pipeline.get_pipeline_metrics()
        assert isinstance(metrics, dict)
        assert metrics["execution_count"] == 1

    def test_multiple_executions(self):
        for _ in range(3):
            self.pipeline.run_complete_pipeline(
                _make_decision()
            )
        metrics = self.pipeline.get_pipeline_metrics()
        assert metrics["execution_count"] == 3
        assert metrics["feedback_count"] == 3

    def test_improvement_cycle_trigger(self):
        config = Phase3Config()
        config.min_samples_for_analysis = 3
        pipeline = Phase3To5Pipeline(config=config)
        for _ in range(5):
            pipeline.run_complete_pipeline(
                _make_decision()
            )
        assert pipeline.should_trigger_improvement_cycle()

    def test_total_latency_recorded(self):
        result = self.pipeline.run_complete_pipeline(
            _make_decision()
        )
        assert result.total_latency_ms >= 0

    def test_phase_latencies_recorded(self):
        result = self.pipeline.run_complete_pipeline(
            _make_decision()
        )
        assert isinstance(result.phase_latencies, dict)

    def test_run_analysis_and_improvement(self):
        config = Phase3Config()
        config.min_samples_for_analysis = 3
        pipeline = Phase3To5Pipeline(config=config)
        for _ in range(5):
            pipeline.run_complete_pipeline(
                _make_decision()
            )
        plan = pipeline.run_analysis_and_improvement()
        # May or may not return a plan depending on data
        assert plan is None or hasattr(plan, "plan_id")

    def test_validation_gate_blocks_action(self):
        pipeline = Phase3To5Pipeline()
        pipeline._validator.analyzer.config.contradiction_threshold = 0.99
        decision = FinalDecisionResult(
            decision="Use existing medical expert",
            confidence=0.9,
            reasoning="",
            action="use_existing",
            expert_name="expert_medical",
            domain="medical",
            metadata={
                "evidence": [
                    "Unrelated evidence about chemistry reactions.",
                ]
            },
        )
        decision.final_decision = "Use existing medical expert"
        pipeline._validator.analyzer.embedding_gen.generate_embedding = (
            lambda text: __import__("numpy").zeros(
                pipeline._validator.analyzer.config.embedding_dim,
                dtype=__import__("numpy").float32,
            )
        )
        pipeline._validator.analyzer.contradiction_detector.embedding_gen = (
            pipeline._validator.analyzer.embedding_gen
        )
        pipeline._validator.analyzer.chain_validator.embedding_gen = (
            pipeline._validator.analyzer.embedding_gen
        )
        decision.reasoning_chain = [
            {"content": "Quantum wavefunctions describe electrons."},
            {"content": "Entropy increases in closed systems."},
        ]
        result = pipeline.run_complete_pipeline(decision)
        assert isinstance(result, SystemExecutionResult)
        assert result.success is False
        assert result.action_result is None
