"""Phase 5.1 – Feedback Integrator.

Integrates feedback and performance analysis into system configuration
updates, expert parameter adjustments, and retraining schedules.
"""

from __future__ import annotations

import copy
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from phase3_validation.config.phase3_config import Phase3Config
from phase3_validation.utils.types import (
    ExpertAnalysis,
    FeedbackData,
    PerformanceAnalysis,
    RetrainingDataset,
    UpdatedExpertConfig,
    UpdatedSystemConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity ordering used by prioritize / sort helpers
# ---------------------------------------------------------------------------
_SEVERITY_RANK: Dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

_RETRAINING_DATA_SCALE: int = 100
_MAX_EXPECTED_IMPROVEMENT: float = 0.5
_IMPROVEMENT_RATE: float = 0.3


class FeedbackIntegrationError(Exception):
    """Raised when feedback integration fails."""


# ===================================================================
# FeedbackIntegrator
# ===================================================================


class FeedbackIntegrator:
    """Integrate feedback into system-wide configuration improvements.

    Consumes a ``PerformanceAnalysis`` together with raw feedback data and
    produces an ``UpdatedSystemConfig`` that describes every change the
    system should apply.
    """

    def __init__(
        self, config: Optional[Phase3Config] = None
    ) -> None:
        """Initialise the integrator.

        Args:
            config: Optional Phase3 configuration.  Falls back to
                defaults when *None*.
        """
        self._config = config or Phase3Config()
        self._integration_history: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def integrate_feedback(
        self,
        performance_analysis: PerformanceAnalysis,
        feedback_data: List[FeedbackData],
    ) -> UpdatedSystemConfig:
        """Build an ``UpdatedSystemConfig`` from analysis and feedback.

        Args:
            performance_analysis: Result of the performance analyser.
            feedback_data: Raw feedback entries collected from users.

        Returns:
            A fully-populated ``UpdatedSystemConfig``.

        Raises:
            FeedbackIntegrationError: If integration fails.
        """
        try:
            start = time.perf_counter()

            updater = ExpertParameterUpdater(self._config)

            # --- expert updates ---
            expert_updates: List[UpdatedExpertConfig] = []
            for expert in performance_analysis.expert_analyses:
                if expert.accuracy < self._config.accuracy_threshold:
                    factors = updater.compute_adjustment_factors(expert)
                    updated = updater.update_expert_parameters(
                        expert.expert_name, expert, factors
                    )
                    expert_updates.append(updated)

            # --- threshold updates ---
            threshold_updates: Dict[str, float] = {}
            health = performance_analysis.system_health_score
            if health < self._config.health_score_threshold:
                new_threshold = round(
                    self._config.accuracy_threshold - 0.05, 4
                )
                threshold_updates["accuracy_threshold"] = new_threshold
                logger.info(
                    "Health %.1f < %.1f – suggesting accuracy_threshold"
                    " adjustment to %.4f",
                    health,
                    self._config.health_score_threshold,
                    new_threshold,
                )

            # --- expert pool changes ---
            expert_pool_changes: Dict[str, Any] = {}
            if performance_analysis.action_required:
                low_domains = [
                    ea.expert_name
                    for ea in performance_analysis.expert_analyses
                    if ea.accuracy < self._config.accuracy_threshold
                ]
                if low_domains:
                    expert_pool_changes["add_experts"] = [
                        {
                            "domain": name,
                            "reason": "low_accuracy",
                        }
                        for name in low_domains
                    ]

            # --- retraining schedule ---
            retraining_schedule: Dict[str, Any] = {}
            for upd in expert_updates:
                if upd.scheduled_retraining:
                    retraining_schedule[upd.expert_name] = {
                        "retraining_timestamp": (
                            upd.retraining_timestamp
                        ),
                        "data_size": upd.retraining_data_size,
                    }

            # --- new features ---
            new_features: List[str] = []
            for issue in performance_analysis.identified_issues:
                desc = issue.get("description", "")
                itype = issue.get("type", "unknown")
                new_features.append(
                    f"address_{itype}: {desc}"
                )

            config_out = UpdatedSystemConfig(
                timestamp=datetime.now(timezone.utc).isoformat(),
                expert_updates=expert_updates,
                threshold_updates=threshold_updates,
                expert_pool_changes=expert_pool_changes,
                retraining_schedule=retraining_schedule,
                new_features_to_add=new_features,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._integration_history.append(
                {
                    "timestamp": config_out.timestamp,
                    "expert_update_count": len(expert_updates),
                    "elapsed_ms": round(elapsed_ms, 2),
                }
            )
            logger.info(
                "Feedback integration completed in %.2f ms "
                "(%d expert updates)",
                elapsed_ms,
                len(expert_updates),
            )
            return config_out

        except Exception as exc:
            raise FeedbackIntegrationError(
                f"Failed to integrate feedback: {exc}"
            ) from exc

    def prioritize_improvements(
        self, issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort issues from highest to lowest severity and rank them.

        Args:
            issues: List of issue dicts, each containing at least a
                ``severity`` key (one of *critical*, *high*, *medium*,
                *low*).

        Returns:
            A new list of issue dicts with an added ``priority_rank``
            field, ordered by descending severity.
        """
        ranked = sorted(
            issues,
            key=lambda i: _SEVERITY_RANK.get(
                i.get("severity", "low"), 0
            ),
            reverse=True,
        )
        for idx, issue in enumerate(ranked, start=1):
            issue["priority_rank"] = idx
        return ranked

    def generate_update_plan(
        self, issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate an ordered action plan from a list of issues.

        Args:
            issues: Prioritised issue list (see
                :meth:`prioritize_improvements`).

        Returns:
            List of update-action dicts, each with *issue_type*,
            *action*, *priority*, and *estimated_effort*.
        """
        plan: List[Dict[str, Any]] = []
        for issue in issues:
            severity = issue.get("severity", "low")
            itype = issue.get("type", "unknown")
            effort = self._estimate_effort(severity)
            plan.append(
                {
                    "issue_type": itype,
                    "action": f"resolve_{itype}",
                    "priority": severity,
                    "estimated_effort": effort,
                }
            )
        return plan

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _estimate_effort(severity: str) -> str:
        """Map severity to a rough effort estimate.

        Args:
            severity: One of *critical*, *high*, *medium*, *low*.

        Returns:
            Effort label string.
        """
        mapping = {
            "critical": "high",
            "high": "medium-high",
            "medium": "medium",
            "low": "low",
        }
        return mapping.get(severity, "medium")


# ===================================================================
# ExpertParameterUpdater
# ===================================================================


class ExpertParameterUpdater:
    """Compute and apply parameter adjustments for individual experts."""

    def __init__(
        self, config: Optional[Phase3Config] = None
    ) -> None:
        """Initialise the updater.

        Args:
            config: Optional Phase3 configuration.
        """
        self._config = config or Phase3Config()
        self._update_history: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def update_expert_parameters(
        self,
        expert_name: str,
        performance_analysis: ExpertAnalysis,
        adjustment_factors: Dict[str, float],
    ) -> UpdatedExpertConfig:
        """Create an ``UpdatedExpertConfig`` for *expert_name*.

        Args:
            expert_name: Identifier of the expert to update.
            performance_analysis: Per-expert analysis data.
            adjustment_factors: Adjustment multipliers produced by
                :meth:`compute_adjustment_factors`.

        Returns:
            Populated ``UpdatedExpertConfig``.
        """
        accuracy = performance_analysis.accuracy
        retraining_data_size = int(
            (1.0 - accuracy) * _RETRAINING_DATA_SCALE
        )
        scheduled = accuracy < self._config.accuracy_threshold
        retraining_ts: Optional[str] = None
        if scheduled:
            future = datetime.now(timezone.utc) + timedelta(hours=24)
            retraining_ts = future.isoformat()

        expected_improvement = min(
            _MAX_EXPECTED_IMPROVEMENT,
            (1.0 - accuracy) * _IMPROVEMENT_RATE,
        )

        updated = UpdatedExpertConfig(
            expert_name=expert_name,
            parameter_updates=dict(adjustment_factors),
            retraining_data_size=retraining_data_size,
            scheduled_retraining=scheduled,
            retraining_timestamp=retraining_ts,
            expected_improvement=round(expected_improvement, 4),
        )

        self._update_history.append(
            {
                "expert": expert_name,
                "accuracy": accuracy,
                "scheduled_retraining": scheduled,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )
        logger.debug(
            "Updated parameters for '%s' (accuracy=%.3f, "
            "scheduled_retraining=%s)",
            expert_name,
            accuracy,
            scheduled,
        )
        return updated

    def compute_adjustment_factors(
        self, analysis: ExpertAnalysis
    ) -> Dict[str, float]:
        """Derive adjustment multipliers from an expert analysis.

        Args:
            analysis: Per-expert analysis result.

        Returns:
            Dict with *learning_rate_factor*,
            *confidence_weight_factor*, and *batch_size_factor*.
        """
        accuracy = analysis.accuracy
        lr_factor = 1.0 + (0.5 - accuracy) * 0.1
        cw_factor = max(
            0.5,
            min(1.5, 1.0 - analysis.confidence_calibration_error),
        )
        bs_factor = 1.0

        return {
            "learning_rate_factor": self._clamp(lr_factor),
            "confidence_weight_factor": self._clamp(cw_factor),
            "batch_size_factor": self._clamp(bs_factor),
        }

    def validate_new_parameters(
        self,
        old_params: Dict[str, Any],
        new_params: Dict[str, Any],
    ) -> bool:
        """Ensure parameter changes stay within allowed bounds.

        Each numeric parameter must not differ from its old value by
        more than ``config.max_parameter_change_pct`` percent.

        Args:
            old_params: Previous parameter values.
            new_params: Proposed parameter values.

        Returns:
            *True* if every change is within the allowed range.
        """
        max_pct = self._config.max_parameter_change_pct
        for key, new_val in new_params.items():
            if key not in old_params:
                continue
            old_val = old_params[key]
            if not isinstance(old_val, (int, float)):
                continue
            if not isinstance(new_val, (int, float)):
                continue
            if old_val == 0:
                continue
            change_pct = abs(new_val - old_val) / abs(old_val) * 100.0
            if change_pct > max_pct:
                logger.warning(
                    "Parameter '%s' change %.1f%% exceeds "
                    "max %.1f%%",
                    key,
                    change_pct,
                    max_pct,
                )
                return False
        return True

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _clamp(
        value: float,
        lo: float = 0.5,
        hi: float = 2.0,
    ) -> float:
        """Clamp *value* to the interval [lo, hi].

        Args:
            value: Number to clamp.
            lo: Lower bound.
            hi: Upper bound.

        Returns:
            Clamped value.
        """
        return max(lo, min(hi, value))


# ===================================================================
# RetrainingDataCollector
# ===================================================================


class RetrainingDataCollector:
    """Collect and prioritise data for expert retraining."""

    def __init__(
        self, config: Optional[Phase3Config] = None
    ) -> None:
        """Initialise the collector.

        Args:
            config: Optional Phase3 configuration.
        """
        self._config = config or Phase3Config()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def collect_retraining_data(
        self,
        feedback_list: List[FeedbackData],
        failure_modes: List[Dict[str, Any]],
    ) -> RetrainingDataset:
        """Build a ``RetrainingDataset`` from feedback and failures.

        Args:
            feedback_list: All available feedback entries.
            failure_modes: Known failure-mode descriptions.

        Returns:
            A ``RetrainingDataset`` ready for consumption.
        """
        incorrect = [
            fb
            for fb in feedback_list
            if fb.prediction_correct is False
        ]

        samples = [
            {
                "input_text": fb.input_text,
                "ground_truth": fb.ground_truth,
                "expert_used": fb.expert_used,
                "confidence": fb.expert_confidence,
            }
            for fb in self.prioritize_samples(feedback_list)
        ]

        failure_samples = [
            {
                "input_text": fb.input_text,
                "ground_truth": fb.ground_truth,
                "expert_used": fb.expert_used,
                "confidence": fb.expert_confidence,
            }
            for fb in incorrect
        ]

        drift = self.detect_data_drift(feedback_list)

        dataset = RetrainingDataset(
            samples=samples,
            failure_samples=failure_samples,
            total_size=len(samples),
            data_drift_detected=drift,
        )

        logger.info(
            "Collected retraining dataset: %d samples, "
            "%d failures, drift=%s",
            dataset.total_size,
            len(failure_samples),
            drift,
        )
        return dataset

    def prioritize_samples(
        self, samples: List[FeedbackData]
    ) -> List[FeedbackData]:
        """Sort feedback so failures come first, then by confidence.

        Args:
            samples: Unsorted feedback entries.

        Returns:
            Sorted copy of *samples*: incorrect predictions first
            (ascending confidence), then correct predictions
            (ascending confidence).
        """
        return sorted(
            samples,
            key=lambda fb: (
                fb.prediction_correct is not False,
                fb.expert_confidence,
            ),
        )

    def detect_data_drift(
        self, new_data: List[FeedbackData]
    ) -> bool:
        """Detect possible data drift with a simple heuristic.

        Compares the accuracy in the most recent 20 % of entries
        against the overall accuracy.  A difference greater than
        10 percentage-points signals drift.

        Args:
            new_data: Feedback entries in chronological order.

        Returns:
            *True* if drift is detected.
        """
        if not new_data:
            return False

        total = len(new_data)
        evaluated = [
            fb for fb in new_data
            if fb.prediction_correct is not None
        ]
        if len(evaluated) < 5:
            return False

        overall_acc = (
            sum(1 for fb in evaluated if fb.prediction_correct)
            / len(evaluated)
        )

        tail_size = max(1, int(total * 0.2))
        tail = new_data[-tail_size:]
        tail_evaluated = [
            fb for fb in tail
            if fb.prediction_correct is not None
        ]
        if not tail_evaluated:
            return False

        tail_acc = (
            sum(1 for fb in tail_evaluated if fb.prediction_correct)
            / len(tail_evaluated)
        )

        drift = abs(overall_acc - tail_acc) > 0.10
        if drift:
            logger.warning(
                "Data drift detected: overall_acc=%.3f, "
                "tail_acc=%.3f",
                overall_acc,
                tail_acc,
            )
        return drift


# ===================================================================
# FeedbackIntegrationPipeline (orchestrator)
# ===================================================================


class FeedbackIntegrationPipeline:
    """Orchestrate the full feedback-integration workflow.

    Creates all sub-components and exposes a single
    :meth:`integrate` entry-point.
    """

    def __init__(
        self, config: Optional[Phase3Config] = None
    ) -> None:
        """Initialise the pipeline and all sub-components.

        Args:
            config: Optional Phase3 configuration.
        """
        self._config = config or Phase3Config()
        self._integrator = FeedbackIntegrator(self._config)
        self._updater = ExpertParameterUpdater(self._config)
        self._collector = RetrainingDataCollector(self._config)
        self._last_metadata: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def integrate(
        self,
        performance_analysis: PerformanceAnalysis,
        feedback_data: Optional[List[FeedbackData]] = None,
    ) -> UpdatedSystemConfig:
        """Run the full integration pipeline.

        Args:
            performance_analysis: Output of the performance analyser.
            feedback_data: Optional raw feedback list.  Defaults to an
                empty list when *None*.

        Returns:
            ``UpdatedSystemConfig`` describing all proposed changes.

        Raises:
            FeedbackIntegrationError: If integration fails.
        """
        if feedback_data is None:
            feedback_data = []

        start = time.perf_counter()
        logger.info(
            "Starting feedback integration pipeline "
            "(%d feedback entries)",
            len(feedback_data),
        )

        result = self._integrator.integrate_feedback(
            performance_analysis, feedback_data
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._last_metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round(elapsed_ms, 2),
            "feedback_count": len(feedback_data),
            "expert_update_count": len(result.expert_updates),
            "threshold_updates": result.threshold_updates,
            "drift_detected": self._collector.detect_data_drift(
                feedback_data
            ),
        }

        logger.info(
            "Integration pipeline completed in %.2f ms",
            elapsed_ms,
        )
        return result

    def get_integration_metadata(self) -> Dict[str, Any]:
        """Return metadata from the last integration run.

        Returns:
            A shallow copy of the most recent metadata dict.
        """
        return copy.copy(self._last_metadata)
