"""Main orchestrator for Phase 3 through Phase 5 execution pipeline.

Receives a FinalDecisionResult from Phase 2.6 and runs the complete
action execution and feedback loop (Phases 3.1 → 4.1 → 4.2 → 5.1 → 5.2).
"""

import logging
import time
from typing import Any, Dict, List, Optional

from phase3_validation.config.phase3_config import Phase3Config
from phase3_validation.phases.phase_3_1_action_executor import (
    ActionExecutionPipeline,
)
from phase3_validation.phases.phase_3_validation import (
    ValidationOrchestrator,
)
from phase3_validation.phases.phase_4_1_feedback_collector import (
    FeedbackCollectionPipeline,
)
from phase3_validation.phases.phase_4_2_performance_analyzer import (
    PerformanceAnalysisPipeline,
)
from phase3_validation.phases.phase_5_1_feedback_integrator import (
    FeedbackIntegrationPipeline,
)
from phase3_validation.phases.phase_5_2_continuous_improvement import (
    ContinuousImprovementPipeline,
)
from phase3_validation.utils.types import (
    ActionResult,
    AggregatedFeedback,
    FeedbackData,
    FinalDecisionResult,
    ImprovementPlan,
    PerformanceAnalysis,
    SystemExecutionResult,
    UpdatedSystemConfig,
)

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the Phase 3-to-5 pipeline encounters a critical error."""


class Phase3To5Pipeline:
    """Complete pipeline orchestrating Phases 3.1 through 5.2.

    Coordinates action execution, feedback collection, performance
    analysis, feedback integration, and continuous improvement in a
    single cohesive workflow.

    Attributes:
        config: Shared configuration for all sub-pipelines.
    """

    def __init__(
        self, config: Optional[Phase3Config] = None
    ) -> None:
        """Initialise the pipeline and all sub-phase pipelines.

        Args:
            config: Optional shared configuration. Uses defaults when
                not provided.
        """
        self.config = config or Phase3Config()

        def _phase2_rerun(text: str, skip_cache: bool = False):
            from phase2_validation.pipeline import Phase2Pipeline

            pipeline = Phase2Pipeline()
            return pipeline.run(text, skip_cache=skip_cache)

        self._action_pipeline = ActionExecutionPipeline(self.config)
        self._validator = ValidationOrchestrator(
            pipeline_fn=_phase2_rerun
        )
        self._feedback_pipeline = FeedbackCollectionPipeline(self.config)
        self._analysis_pipeline = PerformanceAnalysisPipeline(self.config)
        self._integration_pipeline = FeedbackIntegrationPipeline(
            self.config
        )
        self._improvement_pipeline = ContinuousImprovementPipeline(
            self.config
        )

        self._feedback_history: List[FeedbackData] = []
        self._execution_count: int = 0
        self._last_metadata: Dict[str, Any] = {}

        logger.info(
            "Phase3To5Pipeline initialised with min_samples=%d",
            self.config.min_samples_for_analysis,
        )

    def _should_block_action(self, result_class: str) -> bool:
        if not self.config.enforce_validation_gate:
            return False
        allowed = {
            str(item).strip().lower()
            for item in self.config.action_allowed_result_classes
            if str(item).strip()
        }
        if not allowed:
            allowed = {"all_pass"}
        return str(result_class).lower() not in allowed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_complete_pipeline(
        self,
        final_decision_result: FinalDecisionResult,
        ground_truth: Any = None,
        user_rating: Optional[float] = None,
    ) -> SystemExecutionResult:
        """Execute the full Phase 3.1 → 5.2 pipeline for one request.

        Args:
            final_decision_result: Output of Phase 2.6 decision engine.
            ground_truth: Optional ground-truth label for evaluation.
            user_rating: Optional user satisfaction score (0-1).

        Returns:
            SystemExecutionResult containing outputs from every phase
            that was executed.

        Raises:
            PipelineError: If a critical, unrecoverable failure occurs.
        """
        pipeline_start = time.perf_counter()
        phase_latencies: Dict[str, float] = {}
        improvement_plan: Optional[ImprovementPlan] = None

        try:
            # ---- Phase 3 Validation Gate ----
            phase_start = time.perf_counter()
            validation_decision = self._validator.validate(
                final_decision_result
            )

            if (
                validation_decision.result_class == "complete_failure"
                and validation_decision.failure_info
                and validation_decision.failure_info.original_decision
            ):
                final_decision_result = (
                    validation_decision.failure_info.original_decision
                )
            phase_latencies["phase_3_validation"] = (
                (time.perf_counter() - phase_start) * 1000.0
            )

            if self._should_block_action(validation_decision.result_class):
                total_latency_ms = (
                    (time.perf_counter() - pipeline_start) * 1000.0
                )
                error_msg = (
                    "Validation gate blocked action execution: "
                    f"{validation_decision.result_class}"
                )
                logger.warning(error_msg)
                return SystemExecutionResult(
                    original_input=getattr(
                        final_decision_result, "decision", ""
                    ),
                    decision_result=final_decision_result,
                    action_result=None,
                    feedback_data=None,
                    total_latency_ms=total_latency_ms,
                    phase_latencies=phase_latencies,
                    success=False,
                    error_message=error_msg,
                )

            # ---- Phase 3.1: Action Execution ----
            phase_start = time.perf_counter()
            action_result: ActionResult = self._action_pipeline.execute(
                final_decision_result
            )
            phase_latencies["phase_3_1"] = (
                (time.perf_counter() - phase_start) * 1000.0
            )
            logger.info(
                "Phase 3.1 complete: status=%s (%.1f ms)",
                action_result.status,
                phase_latencies["phase_3_1"],
            )

            # ---- Phase 4.1: Feedback Collection ----
            phase_start = time.perf_counter()
            total_latency_ms = sum(phase_latencies.values())
            execution_results: Dict[str, Any] = {
                "input_text": final_decision_result.decision,
                "prediction": action_result.executed_action,
                "ground_truth": ground_truth,
                "user_rating": user_rating,
                "total_latency_ms": total_latency_ms,
                "phase_latencies": dict(phase_latencies),
                "expert_used": final_decision_result.expert_name,
                "expert_confidence": final_decision_result.confidence,
            }
            feedback_data: FeedbackData = (
                self._feedback_pipeline.collect(execution_results)
            )
            phase_latencies["phase_4_1"] = (
                (time.perf_counter() - phase_start) * 1000.0
            )
            logger.info(
                "Phase 4.1 complete: feedback_id=%s (%.1f ms)",
                feedback_data.feedback_id,
                phase_latencies["phase_4_1"],
            )

            self._feedback_history.append(feedback_data)
            self._execution_count += 1

            # ---- Phases 4.2 / 5.1 / 5.2 (conditional) ----
            if self.should_trigger_improvement_cycle():
                phase_start = time.perf_counter()
                improvement_plan = self.run_analysis_and_improvement()
                phase_latencies["phase_4_2_to_5_2"] = (
                    (time.perf_counter() - phase_start) * 1000.0
                )
                logger.info(
                    "Improvement cycle complete (%.1f ms)",
                    phase_latencies["phase_4_2_to_5_2"],
                )

            total_latency_ms = (
                (time.perf_counter() - pipeline_start) * 1000.0
            )

            self._last_metadata = {
                "phase_latencies": phase_latencies,
                "improvement_triggered": improvement_plan is not None,
                "feedback_count": len(self._feedback_history),
                "execution_count": self._execution_count,
                "action_metadata": (
                    self._action_pipeline.get_execution_metadata()
                ),
                "feedback_metadata": (
                    self._feedback_pipeline.get_collection_metadata()
                ),
            }

            return SystemExecutionResult(
                original_input=final_decision_result.decision,
                decision_result=final_decision_result,
                action_result=action_result,
                feedback_data=feedback_data,
                total_latency_ms=total_latency_ms,
                phase_latencies=phase_latencies,
                success=True,
                error_message=None,
            )

        except PipelineError:
            raise
        except Exception as exc:
            total_latency_ms = (
                (time.perf_counter() - pipeline_start) * 1000.0
            )
            error_msg = f"Pipeline failed: {exc}"
            logger.exception(error_msg)
            raise PipelineError(error_msg) from exc

    # ------------------------------------------------------------------
    # Analysis & Improvement
    # ------------------------------------------------------------------

    def run_analysis_and_improvement(
        self,
    ) -> Optional[ImprovementPlan]:
        """Run the analysis and improvement phases (4.2 → 5.1 → 5.2).

        Aggregates accumulated feedback, analyses performance, generates
        updated system configuration, and produces an improvement plan.

        Returns:
            An ImprovementPlan if enough data is available, otherwise
            None.
        """
        if not self.should_trigger_improvement_cycle():
            logger.info(
                "Not enough feedback for analysis (%d/%d required)",
                len(self._feedback_history),
                self.config.min_samples_for_analysis,
            )
            return None

        try:
            # Phase 4.1 aggregate
            aggregated: AggregatedFeedback = (
                self._feedback_pipeline.aggregate(
                    self._feedback_history
                )
            )
            logger.info(
                "Aggregated %d feedback samples (accuracy=%.3f)",
                aggregated.num_samples,
                aggregated.overall_accuracy,
            )

            # Phase 4.2: Performance Analysis
            analysis: PerformanceAnalysis = (
                self._analysis_pipeline.analyze(aggregated)
            )
            logger.info(
                "Performance analysis: health=%.1f, "
                "action_required=%s",
                analysis.system_health_score,
                analysis.action_required,
            )

            # Phase 5.1: Feedback Integration
            updated_config: UpdatedSystemConfig = (
                self._integration_pipeline.integrate(
                    analysis,
                    feedback_data=self._feedback_history,
                )
            )
            logger.info(
                "Integration produced %d expert updates",
                len(updated_config.expert_updates),
            )

            # Phase 5.2: Continuous Improvement
            plan: ImprovementPlan = (
                self._improvement_pipeline.run_improvement_cycle(
                    analysis, updated_config
                )
            )
            logger.info(
                "Improvement plan created: id=%s, effort=%s",
                plan.plan_id,
                plan.estimated_effort,
            )

            self._last_metadata.update({
                "analysis_metadata": (
                    self._analysis_pipeline.get_analysis_metadata()
                ),
                "integration_metadata": (
                    self._integration_pipeline
                    .get_integration_metadata()
                ),
                "improvement_metadata": (
                    self._improvement_pipeline
                    .get_improvement_metadata()
                ),
            })

            return plan

        except Exception as exc:
            logger.exception(
                "Analysis/improvement cycle failed: %s", exc
            )
            raise PipelineError(
                f"Analysis/improvement cycle failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Return aggregate metrics about pipeline execution history.

        Returns:
            Dictionary containing execution counts, feedback counts,
            and per-phase metadata from the most recent run.
        """
        metrics: Dict[str, Any] = {
            "execution_count": self._execution_count,
            "feedback_count": len(self._feedback_history),
            "min_samples_for_analysis": (
                self.config.min_samples_for_analysis
            ),
            "improvement_cycle_ready": (
                self.should_trigger_improvement_cycle()
            ),
            "action_metadata": (
                self._action_pipeline.get_execution_metadata()
            ),
            "feedback_metadata": (
                self._feedback_pipeline.get_collection_metadata()
            ),
            "analysis_metadata": (
                self._analysis_pipeline.get_analysis_metadata()
            ),
            "integration_metadata": (
                self._integration_pipeline
                .get_integration_metadata()
            ),
            "improvement_metadata": (
                self._improvement_pipeline
                .get_improvement_metadata()
            ),
            "last_run_metadata": dict(self._last_metadata),
        }
        return metrics

    def should_trigger_improvement_cycle(self) -> bool:
        """Decide whether an improvement cycle should be triggered.

        Returns:
            True when enough feedback has been collected and at least
            one execution has been recorded.
        """
        has_enough_feedback = (
            len(self._feedback_history)
            >= self.config.min_samples_for_analysis
        )
        has_executions = self._execution_count > 0
        return has_enough_feedback and has_executions
