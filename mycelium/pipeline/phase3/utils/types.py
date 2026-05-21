"""
Data classes for Phases 3-5 (Action Execution, Feedback, Improvement).

Provides typed containers for action results, feedback data,
performance analyses, improvement plans, and system execution
results used throughout the action and feedback pipeline.

Example:
    >>> from phase3_validation.utils.types import (
    ...     ActionResult, FeedbackData, PerformanceAnalysis,
    ... )
    >>> ar = ActionResult(action_type="create_new_expert")
    >>> print(ar.status)
    'pending'
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Phase 2.6 stub type (input to Phase 3.1)
# ------------------------------------------------------------------


@dataclass
class FinalDecisionResult:
    """Output of Phase 2.6 synthesis (stub for downstream use).

    Attributes:
        decision: The final decision text.
        confidence: Overall confidence in ``[0, 1]``.
        reasoning: Explanation of the decision.
        action: Recommended action type.
        expert_name: Name of the selected expert.
        domain: Domain of the decision.
        metadata: Additional decision metadata.
    """

    decision: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    action: str = "use_existing"
    expert_name: str = ""
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Phase 3.1 data classes
# ------------------------------------------------------------------


@dataclass
class CreatedResource:
    """A resource created during action execution.

    Attributes:
        resource_type: Type of resource (expert, patch, dataset).
        resource_id: Unique identifier for the resource.
        domain: Domain the resource belongs to.
        configuration: Resource configuration parameters.
        created_at: ISO-8601 creation timestamp.
        status: Current status (active, pending, failed).
        metadata: Additional resource metadata.
    """

    resource_type: str = ""
    resource_id: str = ""
    domain: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Output of Phase 3.1 action execution.

    Attributes:
        action_type: Type of action executed.
        status: Execution status (success, partial, failed).
        executed_action: Description of the executed action.
        created_resources: Resources created during execution.
        resource_ids: IDs of created resources.
        execution_time_ms: Wall-clock execution time.
        error_message: Error details if failed.
        rollback_available: Whether rollback is possible.
        metadata: Additional execution metadata.
        warnings: Non-fatal issues encountered.
    """

    action_type: str = ""
    status: str = "pending"
    executed_action: str = ""
    created_resources: List[CreatedResource] = field(
        default_factory=list
    )
    resource_ids: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    rollback_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Phase 4.1 data classes
# ------------------------------------------------------------------


@dataclass
class FeedbackData:
    """Collected feedback from a single execution.

    Attributes:
        input_text: Original input text.
        prediction: The system's prediction/decision.
        ground_truth: Correct answer if available.
        prediction_correct: Whether prediction matched truth.
        user_rating: User satisfaction rating (1-5).
        total_latency_ms: Total pipeline latency.
        phase_latencies: Per-phase latency breakdown.
        expert_used: Name of the expert used.
        expert_confidence: Confidence of the expert.
        timestamp: ISO-8601 timestamp.
        session_id: Session identifier.
        feedback_id: Unique feedback identifier.
    """

    input_text: str = ""
    prediction: Any = None
    ground_truth: Any = None
    prediction_correct: Optional[bool] = None
    user_rating: Optional[float] = None
    total_latency_ms: float = 0.0
    phase_latencies: Dict[str, float] = field(
        default_factory=dict
    )
    expert_used: str = ""
    expert_confidence: float = 0.0
    timestamp: str = ""
    session_id: str = ""
    feedback_id: str = ""


@dataclass
class AggregatedFeedback:
    """Aggregated feedback metrics over multiple executions.

    Attributes:
        time_period: Period covered (e.g. '1h', '1d').
        num_samples: Number of feedback entries.
        overall_accuracy: Accuracy across all predictions.
        overall_calibration_error: Calibration error metric.
        average_latency_ms: Mean latency.
        expert_performance: Per-expert performance metrics.
        user_satisfaction: Average user rating.
        common_failure_modes: Patterns in failures.
        trends: Metric trends (improving, stable, declining).
    """

    time_period: str = ""
    num_samples: int = 0
    overall_accuracy: float = 0.0
    overall_calibration_error: float = 0.0
    average_latency_ms: float = 0.0
    expert_performance: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    user_satisfaction: float = 0.0
    common_failure_modes: List[Dict[str, Any]] = field(
        default_factory=list
    )
    trends: Dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------
# Phase 4.2 data classes
# ------------------------------------------------------------------


@dataclass
class ExpertAnalysis:
    """Analysis of a single expert's performance.

    Attributes:
        expert_name: Expert identifier.
        total_samples: Number of samples evaluated.
        accuracy: Prediction accuracy in ``[0, 1]``.
        precision: Precision metric in ``[0, 1]``.
        recall: Recall metric in ``[0, 1]``.
        f1_score: F1 score in ``[0, 1]``.
        average_latency_ms: Mean latency for this expert.
        confidence_calibration_error: Calibration error.
        reliability_score: Overall reliability in ``[0, 1]``.
        performance_trend: Trend direction string.
        last_updated: ISO-8601 timestamp.
    """

    expert_name: str = ""
    total_samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    average_latency_ms: float = 0.0
    confidence_calibration_error: float = 0.0
    reliability_score: float = 0.0
    performance_trend: str = "stable"
    last_updated: str = ""


@dataclass
class PerformanceAnalysis:
    """System-wide performance analysis.

    Attributes:
        analysis_timestamp: When analysis was performed.
        time_period: Period analyzed.
        overall_accuracy: System-wide accuracy.
        overall_latency_ms: System-wide mean latency.
        expert_analyses: Per-expert analysis results.
        identified_issues: List of identified problems.
        issue_severity_scores: Severity per issue.
        improvement_recommendations: Suggested improvements.
        system_health_score: Overall health score (0-100).
        action_required: Whether action is needed.
    """

    analysis_timestamp: str = ""
    time_period: str = ""
    overall_accuracy: float = 0.0
    overall_latency_ms: float = 0.0
    expert_analyses: List[ExpertAnalysis] = field(
        default_factory=list
    )
    identified_issues: List[Dict[str, Any]] = field(
        default_factory=list
    )
    issue_severity_scores: Dict[str, float] = field(
        default_factory=dict
    )
    improvement_recommendations: List[str] = field(
        default_factory=list
    )
    system_health_score: float = 100.0
    action_required: bool = False


# ------------------------------------------------------------------
# Phase 5.1 data classes
# ------------------------------------------------------------------


@dataclass
class UpdatedExpertConfig:
    """Updated configuration for a single expert.

    Attributes:
        expert_name: Expert identifier.
        parameter_updates: Parameter changes to apply.
        retraining_data_size: Number of retraining samples.
        scheduled_retraining: Whether retraining is scheduled.
        retraining_timestamp: When retraining will occur.
        expected_improvement: Expected improvement factor.
    """

    expert_name: str = ""
    parameter_updates: Dict[str, Any] = field(
        default_factory=dict
    )
    retraining_data_size: int = 0
    scheduled_retraining: bool = False
    retraining_timestamp: Optional[str] = None
    expected_improvement: float = 0.0


@dataclass
class UpdatedSystemConfig:
    """System configuration updates from feedback integration.

    Attributes:
        timestamp: When updates were generated.
        expert_updates: Per-expert configuration updates.
        threshold_updates: Updated threshold values.
        expert_pool_changes: Changes to expert pool.
        retraining_schedule: Schedule for retraining.
        new_features_to_add: Features to add.
    """

    timestamp: str = ""
    expert_updates: List[UpdatedExpertConfig] = field(
        default_factory=list
    )
    threshold_updates: Dict[str, float] = field(
        default_factory=dict
    )
    expert_pool_changes: Dict[str, Any] = field(
        default_factory=dict
    )
    retraining_schedule: Dict[str, Any] = field(
        default_factory=dict
    )
    new_features_to_add: List[str] = field(
        default_factory=list
    )


@dataclass
class RetrainingDataset:
    """Dataset collected for expert retraining.

    Attributes:
        expert_name: Target expert for retraining.
        samples: Feedback samples for retraining.
        failure_samples: Samples from failure cases.
        total_size: Total number of samples.
        data_drift_detected: Whether distribution shift found.
        priority_scores: Per-sample priority scores.
    """

    expert_name: str = ""
    samples: List[Dict[str, Any]] = field(
        default_factory=list
    )
    failure_samples: List[Dict[str, Any]] = field(
        default_factory=list
    )
    total_size: int = 0
    data_drift_detected: bool = False
    priority_scores: List[float] = field(default_factory=list)


# ------------------------------------------------------------------
# Phase 5.2 data classes
# ------------------------------------------------------------------


@dataclass
class ImprovementPlan:
    """Plan for system improvements.

    Attributes:
        plan_id: Unique plan identifier.
        timestamp: When the plan was created.
        actions: List of improvement actions.
        expected_improvements: Expected metric improvements.
        timeline: Action timeline.
        success_metrics: Metrics to measure success.
        estimated_effort: Effort level (low, medium, high).
    """

    plan_id: str = ""
    timestamp: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    expected_improvements: Dict[str, float] = field(
        default_factory=dict
    )
    timeline: Dict[str, str] = field(default_factory=dict)
    success_metrics: Dict[str, float] = field(
        default_factory=dict
    )
    estimated_effort: str = "medium"


@dataclass
class ImprovementProgress:
    """Progress of an improvement cycle.

    Attributes:
        plan_id: Associated plan identifier.
        start_timestamp: When improvement started.
        current_timestamp: Current time.
        progress_percentage: Completion percentage.
        completed_actions: Actions completed so far.
        pending_actions: Actions remaining.
        achieved_improvements: Improvements achieved so far.
        status: Current status string.
    """

    plan_id: str = ""
    start_timestamp: str = ""
    current_timestamp: str = ""
    progress_percentage: float = 0.0
    completed_actions: List[str] = field(default_factory=list)
    pending_actions: List[str] = field(default_factory=list)
    achieved_improvements: Dict[str, float] = field(
        default_factory=dict
    )
    status: str = "in_progress"


# ------------------------------------------------------------------
# System-wide result
# ------------------------------------------------------------------


@dataclass
class SystemExecutionResult:
    """Complete system execution result (Phase 2.1 to 5.2).

    Attributes:
        original_input: Raw user input.
        decision_result: Phase 2.6 output.
        action_result: Phase 3.1 output.
        feedback_data: Phase 4.1 output.
        total_latency_ms: Total pipeline latency.
        phase_latencies: Per-phase latency breakdown.
        success: Whether execution succeeded.
        error_message: Error details if failed.
    """

    original_input: str = ""
    decision_result: Optional[FinalDecisionResult] = None
    action_result: Optional[ActionResult] = None
    feedback_data: Optional[FeedbackData] = None
    total_latency_ms: float = 0.0
    phase_latencies: Dict[str, float] = field(
        default_factory=dict
    )
    success: bool = False
    error_message: Optional[str] = None
