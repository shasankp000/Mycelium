"""Phase 5.2 – Continuous Improvement Loop.

Orchestrates a continuous improvement cycle for the Mycelium expert
system.  The module provides four cooperating components:

* **ContinuousImprovementLoop** – creates and tracks improvement plans.
* **ImprovementExecutor** – simulates applying expert updates and
  retraining.
* **ImprovementMetricsTracker** – records per-metric history and
  extrapolates future performance.
* **ContinuousImprovementPipeline** – top-level orchestrator that ties
  the other three components together.
"""

from __future__ import annotations

import copy
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from phase3_validation.config.phase3_config import Phase3Config
from phase3_validation.utils.types import (
    ImprovementPlan,
    ImprovementProgress,
    PerformanceAnalysis,
    UpdatedExpertConfig,
    UpdatedSystemConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEGRADATION_TOLERANCE: float = 0.05
"""Maximum relative degradation (5 %) before a metric is flagged."""

_LOW_EFFORT_THRESHOLD: int = 3
"""Actions at or below this count are considered *low* effort."""

_MEDIUM_EFFORT_THRESHOLD: int = 7
"""Actions at or below this count are considered *medium* effort."""

_DEFAULT_CONFIDENCE: float = 0.95
"""Starting confidence for performance predictions."""

_CONFIDENCE_DECAY: float = 0.05
"""Per-step confidence decay for predictions."""


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ContinuousImprovementError(Exception):
    """Raised when a continuous-improvement operation fails."""


# ---------------------------------------------------------------------------
# ContinuousImprovementLoop
# ---------------------------------------------------------------------------


class ContinuousImprovementLoop:
    """Create and track improvement plans based on performance feedback.

    Each call to :meth:`execute_improvement_cycle` produces an
    ``ImprovementPlan`` that is stored in the internal cycle history
    for later inspection.
    """

    def __init__(
        self, config: Optional[Phase3Config] = None
    ) -> None:
        """Initialise the improvement loop.

        Args:
            config: Optional ``Phase3Config``.  Uses defaults when
                ``None``.
        """
        self._config = config or Phase3Config()
        self._cycle_history: List[ImprovementPlan] = []
        self._current_plan: Optional[ImprovementPlan] = None

    # -- public API --------------------------------------------------------

    def execute_improvement_cycle(
        self,
        performance_analysis: PerformanceAnalysis,
        system_config: UpdatedSystemConfig,
    ) -> ImprovementPlan:
        """Build an improvement plan from analysis and system config.

        For every expert update in *system_config* an ``expert_update``
        action is created.  Each recommendation string in
        *performance_analysis* becomes a ``recommendation`` action.

        Args:
            performance_analysis: Latest ``PerformanceAnalysis``.
            system_config: ``UpdatedSystemConfig`` with expert updates.

        Returns:
            A fully-populated ``ImprovementPlan``.

        Raises:
            ContinuousImprovementError: If plan creation fails.
        """
        start = time.perf_counter()
        logger.info(
            "Starting improvement cycle (experts=%d, recs=%d)",
            len(system_config.expert_updates),
            len(performance_analysis.improvement_recommendations),
        )

        try:
            actions = self._build_actions(
                system_config.expert_updates,
                performance_analysis.improvement_recommendations,
            )
            expected_improvements = (
                self._estimate_improvements(
                    system_config.expert_updates
                )
            )
            timeline = self._build_timeline(
                system_config.expert_updates
            )
            success_metrics = self._build_success_metrics()
            estimated_effort = self._classify_effort(len(actions))

            plan = ImprovementPlan(
                plan_id=f"plan_{uuid4().hex[:12]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                actions=actions,
                expected_improvements=expected_improvements,
                timeline=timeline,
                success_metrics=success_metrics,
                estimated_effort=estimated_effort,
            )

            self._current_plan = plan
            self._cycle_history.append(plan)

            return plan

        except ContinuousImprovementError:
            raise
        except Exception as exc:
            raise ContinuousImprovementError(
                f"Improvement cycle failed: {exc}"
            ) from exc
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "Improvement cycle complete in %.2f ms", elapsed_ms
            )

    def schedule_next_cycle(
        self, current_config: Dict[str, Any]
    ) -> str:
        """Return an ISO-8601 timestamp for the next cycle.

        The interval is taken from
        ``Phase3Config.improvement_cycle_interval_hours``.

        Args:
            current_config: Current configuration dictionary
                (reserved for future scheduling heuristics).

        Returns:
            ISO-8601 timestamp of the next scheduled cycle.
        """
        interval_hours = (
            self._config.improvement_cycle_interval_hours
        )
        next_time = datetime.now(timezone.utc) + timedelta(
            hours=interval_hours
        )
        next_ts = next_time.isoformat()
        logger.info(
            "Next improvement cycle scheduled at %s "
            "(interval=%d h)",
            next_ts,
            interval_hours,
        )
        return next_ts

    def track_improvement_progress(
        self,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
    ) -> ImprovementProgress:
        """Compare old and new metrics to track progress.

        Metrics that improved go into ``completed_actions``; those that
        did not are ``pending_actions``.  Progress percentage is
        computed as the fraction of metrics that improved relative to
        the total number of tracked metrics.

        Args:
            old_metrics: Metric values before the improvement cycle.
            new_metrics: Metric values after the improvement cycle.

        Returns:
            An ``ImprovementProgress`` snapshot.
        """
        all_keys = sorted(
            set(old_metrics.keys()) | set(new_metrics.keys())
        )

        completed: List[str] = []
        pending: List[str] = []
        achieved: Dict[str, float] = {}

        for key in all_keys:
            old_val = old_metrics.get(key, 0.0)
            new_val = new_metrics.get(key, 0.0)
            delta = new_val - old_val
            achieved[key] = round(delta, 6)

            if delta > 0:
                completed.append(key)
            else:
                pending.append(key)

        total = len(all_keys) if all_keys else 1
        progress_pct = (len(completed) / total) * 100.0

        plan_id = (
            self._current_plan.plan_id
            if self._current_plan
            else ""
        )

        now_ts = datetime.now(timezone.utc).isoformat()
        start_ts = (
            self._current_plan.timestamp
            if self._current_plan
            else now_ts
        )

        status = "completed" if not pending else "in_progress"

        progress = ImprovementProgress(
            plan_id=plan_id,
            start_timestamp=start_ts,
            current_timestamp=now_ts,
            progress_percentage=round(progress_pct, 2),
            completed_actions=completed,
            pending_actions=pending,
            achieved_improvements=achieved,
            status=status,
        )
        logger.info(
            "Improvement progress: %.1f %% (%d/%d metrics improved)",
            progress_pct,
            len(completed),
            total,
        )
        return progress

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _build_actions(
        expert_updates: List[UpdatedExpertConfig],
        recommendations: List[str],
    ) -> List[Dict[str, Any]]:
        """Compile a flat list of action dicts."""
        actions: List[Dict[str, Any]] = []
        for update in expert_updates:
            actions.append(
                {
                    "type": "expert_update",
                    "expert_name": update.expert_name,
                    "parameters": update.parameter_updates,
                }
            )
        for rec in recommendations:
            actions.append(
                {
                    "type": "recommendation",
                    "description": rec,
                }
            )
        return actions

    @staticmethod
    def _estimate_improvements(
        expert_updates: List[UpdatedExpertConfig],
    ) -> Dict[str, float]:
        """Map each expert to its expected accuracy improvement."""
        improvements: Dict[str, float] = {}
        for update in expert_updates:
            improvements[update.expert_name] = round(
                update.expected_improvement, 4
            )
        return improvements

    @staticmethod
    def _build_timeline(
        expert_updates: List[UpdatedExpertConfig],
    ) -> Dict[str, str]:
        """Classify actions as *immediate* or *scheduled*."""
        timeline: Dict[str, str] = {}
        for update in expert_updates:
            if update.scheduled_retraining:
                timeline[update.expert_name] = "scheduled"
            else:
                timeline[update.expert_name] = "immediate"
        return timeline

    def _build_success_metrics(self) -> Dict[str, float]:
        """Derive success metrics from configuration thresholds."""
        return {
            "target_accuracy": self._config.accuracy_threshold,
            "target_latency_ms": (
                self._config.latency_threshold_ms
            ),
        }

    @staticmethod
    def _classify_effort(action_count: int) -> str:
        """Classify the effort level for a given action count."""
        if action_count <= _LOW_EFFORT_THRESHOLD:
            return "low"
        if action_count <= _MEDIUM_EFFORT_THRESHOLD:
            return "medium"
        return "high"


# ---------------------------------------------------------------------------
# ImprovementExecutor
# ---------------------------------------------------------------------------


class ImprovementExecutor:
    """Simulate execution of improvement actions.

    Each public method returns a summary dict and appends an entry to
    the internal execution log.
    """

    def __init__(
        self, config: Optional[Phase3Config] = None
    ) -> None:
        """Initialise the executor.

        Args:
            config: Optional ``Phase3Config``.  Uses defaults when
                ``None``.
        """
        self._config = config or Phase3Config()
        self._execution_log: List[Dict[str, Any]] = []

    # -- public API --------------------------------------------------------

    def apply_expert_updates(
        self, expert_updates: List[UpdatedExpertConfig]
    ) -> Dict[str, Any]:
        """Simulate applying parameter updates to experts.

        Args:
            expert_updates: List of ``UpdatedExpertConfig`` to apply.

        Returns:
            Dict with ``applied_count``, ``failed_count`` and
            ``details`` (per-expert result list).
        """
        start = time.perf_counter()
        logger.info(
            "Applying expert updates (count=%d)",
            len(expert_updates),
        )

        details: List[Dict[str, Any]] = []
        applied = 0
        failed = 0

        for update in expert_updates:
            try:
                result = self._apply_single_update(update)
                details.append(result)
                if result["status"] == "applied":
                    applied += 1
                else:
                    failed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to apply update for '%s': %s",
                    update.expert_name,
                    exc,
                )
                details.append(
                    {
                        "expert_name": update.expert_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                failed += 1

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        summary: Dict[str, Any] = {
            "applied_count": applied,
            "failed_count": failed,
            "details": details,
        }

        self._execution_log.append(
            {
                "action": "apply_expert_updates",
                "timestamp": (
                    datetime.now(timezone.utc).isoformat()
                ),
                "summary": summary,
                "elapsed_ms": round(elapsed_ms, 2),
            }
        )
        logger.info(
            "Expert updates applied in %.2f ms "
            "(applied=%d, failed=%d)",
            elapsed_ms,
            applied,
            failed,
        )
        return summary

    def retrain_experts(
        self, retraining_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simulate scheduling expert retraining jobs.

        Args:
            retraining_configs: List of dicts, each containing at
                least ``expert_name``.

        Returns:
            Dict with ``scheduled_count``, ``expert_names`` and
            ``estimated_completion`` ISO-8601 timestamp.
        """
        start = time.perf_counter()
        logger.info(
            "Scheduling retraining (count=%d)",
            len(retraining_configs),
        )

        expert_names: List[str] = []
        for cfg in retraining_configs:
            name = cfg.get("expert_name", "unknown")
            expert_names.append(name)
            logger.debug("Scheduled retraining for '%s'", name)

        estimated_completion = (
            datetime.now(timezone.utc) + timedelta(hours=24)
        ).isoformat()

        summary: Dict[str, Any] = {
            "scheduled_count": len(retraining_configs),
            "expert_names": expert_names,
            "estimated_completion": estimated_completion,
        }

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._execution_log.append(
            {
                "action": "retrain_experts",
                "timestamp": (
                    datetime.now(timezone.utc).isoformat()
                ),
                "summary": summary,
                "elapsed_ms": round(elapsed_ms, 2),
            }
        )
        logger.info(
            "Retraining scheduled in %.2f ms "
            "(experts=%s)",
            elapsed_ms,
            expert_names,
        )
        return summary

    def verify_improvements(
        self,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float],
    ) -> bool:
        """Check that at least one metric improved without degradation.

        A metric is considered *significantly degraded* when the new
        value is more than 5 % worse than the old value.

        Args:
            before_metrics: Baseline metric values.
            after_metrics: Post-improvement metric values.

        Returns:
            ``True`` if at least one metric improved and none degraded
            significantly.
        """
        any_improved = False
        for key in before_metrics:
            if key not in after_metrics:
                continue
            old_val = before_metrics[key]
            new_val = after_metrics[key]
            delta = new_val - old_val

            if delta > 0:
                any_improved = True

            if old_val > 0 and delta < 0:
                relative_drop = abs(delta) / old_val
                if relative_drop > _DEGRADATION_TOLERANCE:
                    logger.warning(
                        "Metric '%s' degraded by %.1f %%",
                        key,
                        relative_drop * 100.0,
                    )
                    return False

        if not any_improved:
            logger.info("No metrics improved")
        return any_improved

    # -- internal helpers --------------------------------------------------

    def _apply_single_update(
        self, update: UpdatedExpertConfig
    ) -> Dict[str, Any]:
        """Apply a single expert parameter update (simulated)."""
        logger.debug(
            "Applying update for expert '%s' "
            "(params=%d, expected_improvement=%.4f)",
            update.expert_name,
            len(update.parameter_updates),
            update.expected_improvement,
        )
        return {
            "expert_name": update.expert_name,
            "status": "applied",
            "parameters_updated": list(
                update.parameter_updates.keys()
            ),
            "expected_improvement": update.expected_improvement,
        }


# ---------------------------------------------------------------------------
# ImprovementMetricsTracker
# ---------------------------------------------------------------------------


class ImprovementMetricsTracker:
    """Track per-metric history and extrapolate future performance."""

    def __init__(self) -> None:
        """Initialise the tracker with an empty metrics store."""
        self._metrics: Dict[
            str, List[Tuple[str, float]]
        ] = {}

    # -- public API --------------------------------------------------------

    def track_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a metric observation.

        Args:
            metric_name: Name of the metric.
            value: Observed value.
            timestamp: Optional ISO-8601 timestamp.  Uses *now* when
                ``None``.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        self._metrics.setdefault(metric_name, []).append(
            (ts, value)
        )
        logger.debug(
            "Tracked metric '%s' = %.6f at %s",
            metric_name,
            value,
            ts,
        )

    def compute_improvement_velocity(
        self, metric_name: str
    ) -> float:
        """Compute the average rate of change per data point.

        Args:
            metric_name: Name of the metric.

        Returns:
            ``(last - first) / num_points`` or ``0.0`` when fewer
            than two observations exist.
        """
        entries = self._metrics.get(metric_name, [])
        if len(entries) < 2:
            return 0.0

        first_val = entries[0][1]
        last_val = entries[-1][1]
        velocity = (last_val - first_val) / len(entries)
        return velocity

    def predict_future_performance(
        self, metric_name: str, steps: int = 5
    ) -> Dict[str, Any]:
        """Extrapolate future metric values linearly.

        Confidence decreases by ``_CONFIDENCE_DECAY`` for each
        prediction step.

        Args:
            metric_name: Name of the metric.
            steps: Number of steps to project.

        Returns:
            Dict with ``current_value``, ``predicted_value``,
            ``velocity``, ``steps`` and ``confidence``.
        """
        entries = self._metrics.get(metric_name, [])
        current_value = entries[-1][1] if entries else 0.0
        velocity = self.compute_improvement_velocity(metric_name)
        predicted_value = current_value + velocity * steps
        confidence = max(
            0.0,
            _DEFAULT_CONFIDENCE - _CONFIDENCE_DECAY * steps,
        )

        return {
            "current_value": round(current_value, 6),
            "predicted_value": round(predicted_value, 6),
            "velocity": round(velocity, 6),
            "steps": steps,
            "confidence": round(confidence, 4),
        }

    def get_metric_history(
        self, metric_name: str
    ) -> List[float]:
        """Return the recorded values for a metric (no timestamps).

        Args:
            metric_name: Name of the metric.

        Returns:
            List of float values in recording order.
        """
        return [
            v for _, v in self._metrics.get(metric_name, [])
        ]


# ---------------------------------------------------------------------------
# ContinuousImprovementPipeline
# ---------------------------------------------------------------------------


class ContinuousImprovementPipeline:
    """Orchestrate the full continuous-improvement workflow.

    Ties together ``ContinuousImprovementLoop``,
    ``ImprovementExecutor`` and ``ImprovementMetricsTracker`` into a
    single entry-point.
    """

    def __init__(
        self, config: Optional[Phase3Config] = None
    ) -> None:
        """Initialise the pipeline.

        Args:
            config: Optional ``Phase3Config``.  Uses defaults when
                ``None``.
        """
        self._config = config or Phase3Config()
        self._loop = ContinuousImprovementLoop(self._config)
        self._executor = ImprovementExecutor(self._config)
        self._tracker = ImprovementMetricsTracker()
        self._last_metadata: Dict[str, Any] = {}

        errors = self._config.validate()
        if errors:
            logger.warning(
                "Config validation errors: %s", errors
            )

    # -- public API --------------------------------------------------------

    def run_improvement_cycle(
        self,
        performance_analysis: PerformanceAnalysis,
        system_config: UpdatedSystemConfig,
    ) -> ImprovementPlan:
        """Execute a full improvement cycle.

        1. Creates an improvement plan via the loop.
        2. Applies expert updates via the executor.
        3. Records key metrics in the tracker.

        Args:
            performance_analysis: Current ``PerformanceAnalysis``.
            system_config: ``UpdatedSystemConfig`` with expert
                updates.

        Returns:
            The ``ImprovementPlan`` produced by the cycle.

        Raises:
            ContinuousImprovementError: If the cycle fails.
        """
        start = time.perf_counter()
        logger.info("Pipeline improvement cycle starting")

        try:
            plan = self._loop.execute_improvement_cycle(
                performance_analysis, system_config
            )

            if system_config.expert_updates:
                self._executor.apply_expert_updates(
                    system_config.expert_updates
                )

            self._record_cycle_metrics(
                performance_analysis, plan
            )

            return plan

        except ContinuousImprovementError:
            raise
        except Exception as exc:
            raise ContinuousImprovementError(
                f"Pipeline improvement cycle failed: {exc}"
            ) from exc
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._last_metadata = {
                "timestamp": (
                    datetime.now(timezone.utc).isoformat()
                ),
                "elapsed_ms": round(elapsed_ms, 2),
                "plan_id": (
                    self._loop._current_plan.plan_id
                    if self._loop._current_plan
                    else None
                ),
            }
            if self._config.enable_performance_logging:
                logger.info(
                    "Pipeline cycle finished in %.2f ms",
                    elapsed_ms,
                )

    def get_improvement_status(self) -> Dict[str, Any]:
        """Return a summary of current improvement state.

        Returns:
            Dict with ``current_plan``, ``cycle_count`` and
            ``metrics_tracked``.
        """
        current = self._loop._current_plan
        return {
            "current_plan": (
                current.plan_id if current else None
            ),
            "cycle_count": len(self._loop._cycle_history),
            "metrics_tracked": list(
                self._tracker._metrics.keys()
            ),
        }

    def get_improvement_metadata(self) -> Dict[str, Any]:
        """Return metadata from the last pipeline run.

        Returns:
            A shallow copy of the metadata dictionary.
        """
        return copy.copy(self._last_metadata)

    # -- internal helpers --------------------------------------------------

    def _record_cycle_metrics(
        self,
        analysis: PerformanceAnalysis,
        plan: ImprovementPlan,
    ) -> None:
        """Persist key metrics from the cycle into the tracker."""
        ts = datetime.now(timezone.utc).isoformat()
        self._tracker.track_metric(
            "overall_accuracy",
            analysis.overall_accuracy,
            timestamp=ts,
        )
        self._tracker.track_metric(
            "overall_latency_ms",
            analysis.overall_latency_ms,
            timestamp=ts,
        )
        self._tracker.track_metric(
            "system_health_score",
            analysis.system_health_score,
            timestamp=ts,
        )
        self._tracker.track_metric(
            "action_count",
            float(len(plan.actions)),
            timestamp=ts,
        )
