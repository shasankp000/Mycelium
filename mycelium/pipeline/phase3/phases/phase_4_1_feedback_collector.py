"""Phase 4.1 – Feedback Collector.

Collects performance feedback after action execution, computes
accuracy / latency / calibration metrics, and aggregates results
for the improvement loop.

Example::

    pipeline = FeedbackCollectionPipeline()
    fb = pipeline.collect({
        "input_text": "query", "prediction": "answer",
        "ground_truth": "answer",
        "execution_metrics": {"total_latency_ms": 42.0},
    })
    agg = pipeline.aggregate([fb])
"""

from __future__ import annotations

import copy
import logging
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from phase3_validation.config.phase3_config import Phase3Config
from phase3_validation.utils.types import (
    AggregatedFeedback,
    FeedbackData,
)

logger = logging.getLogger(__name__)


class FeedbackCollectionError(Exception):
    """Raised when feedback collection or aggregation fails."""


class FeedbackCollector:
    """Collect performance feedback from individual executions."""

    def __init__(self, config: Optional[Phase3Config] = None) -> None:
        """Initialise the collector.

        Args:
            config: Optional configuration; defaults used when *None*.
        """
        self._config = config or Phase3Config()
        self._feedback_history: List[FeedbackData] = []
        self._latency_records: Dict[str, List[float]] = {}
        self._expert_contributions: Dict[str, List[float]] = {}
        self._user_feedback_log: List[Dict[str, Any]] = []

    def collect_feedback(
        self,
        input_text: str,
        prediction: Any,
        ground_truth: Any = None,
        user_rating: Optional[float] = None,
        execution_metrics: Optional[Dict[str, Any]] = None,
    ) -> FeedbackData:
        """Create a :class:`FeedbackData` entry from execution results.

        Args:
            input_text: The original user query.
            prediction: Model prediction / action output.
            ground_truth: Expected correct value, if available.
            user_rating: Optional satisfaction score.
            execution_metrics: Optional dict with timing / expert info.

        Returns:
            A populated :class:`FeedbackData` instance.

        Raises:
            FeedbackCollectionError: If an unexpected error occurs.
        """
        start = time.perf_counter()
        try:
            metrics = execution_metrics or {}

            prediction_correct: Optional[bool] = None
            if ground_truth is not None:
                prediction_correct = prediction == ground_truth

            feedback = FeedbackData(
                input_text=input_text,
                prediction=prediction,
                ground_truth=ground_truth,
                prediction_correct=prediction_correct,
                user_rating=user_rating,
                total_latency_ms=metrics.get("total_latency_ms", 0.0),
                phase_latencies=metrics.get("phase_latencies", {}),
                expert_used=metrics.get("expert_used", ""),
                expert_confidence=metrics.get("expert_confidence", 0.0),
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=f"session_{uuid4().hex[:8]}",
                feedback_id=f"fb_{uuid4().hex[:12]}",
            )

            self._feedback_history.append(feedback)
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.debug(
                "Collected feedback %s in %.2f ms",
                feedback.feedback_id, elapsed,
            )
            return feedback

        except Exception as exc:
            raise FeedbackCollectionError(
                f"Failed to collect feedback: {exc}") from exc

    def record_prediction_latency(
        self, phase: str, latency_ms: float,
    ) -> None:
        """Append a latency measurement for *phase*."""
        self._latency_records.setdefault(phase, []).append(latency_ms)
        logger.debug("Recorded latency for %s: %.2f ms", phase, latency_ms)

    def record_user_feedback(self, rating: float, comments: str) -> None:
        """Store a user-provided feedback entry."""
        self._user_feedback_log.append({
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.debug("Recorded user feedback: rating=%.2f", rating)

    def record_expert_contribution(
        self, expert_name: str, confidence: float,
    ) -> None:
        """Record an expert's confidence for a single prediction."""
        self._expert_contributions.setdefault(expert_name, []).append(
            confidence,
        )
        logger.debug(
            "Recorded expert contribution: %s confidence=%.4f",
            expert_name, confidence,
        )


class PerformanceMetricsCompute:
    """Compute aggregate performance metrics from feedback data."""

    def __init__(self) -> None:
        """Initialise the metrics computer."""

    def compute_accuracy(
        self, predictions: List[Any], ground_truth: List[Any],
    ) -> float:
        """Return fraction of matching prediction / ground-truth pairs.

        Args:
            predictions: Model predictions.
            ground_truth: Expected correct values.

        Returns:
            Accuracy in [0.0, 1.0], or 0.0 if lists are empty.
        """
        if not predictions or not ground_truth:
            return 0.0
        count = min(len(predictions), len(ground_truth))
        correct = sum(
            1 for p, g in zip(predictions[:count], ground_truth[:count])
            if p == g
        )
        return correct / count

    def compute_calibration_error(
        self, predicted_confidence: List[float], accuracies: List[bool],
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Uses 10 equally-spaced confidence bins, weighted by bin size.

        Args:
            predicted_confidence: Per-sample confidence values.
            accuracies: Per-sample correctness flags.

        Returns:
            ECE value, or 0.0 when inputs are empty.
        """
        if not predicted_confidence or not accuracies:
            return 0.0

        n_bins = 10
        total = len(predicted_confidence)
        ece = 0.0

        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            indices = [
                i for i, c in enumerate(predicted_confidence)
                if lo <= c < hi or (b == n_bins - 1 and c == hi)
            ]
            if not indices:
                continue

            bin_conf = statistics.mean(
                predicted_confidence[i] for i in indices)
            bin_acc = statistics.mean(
                float(accuracies[i]) for i in indices)
            ece += (len(indices) / total) * abs(bin_conf - bin_acc)

        return ece

    def compute_latency_metrics(
        self, latencies: Dict[str, float],
    ) -> Dict[str, Any]:
        """Derive summary statistics from per-phase latency values.

        Args:
            latencies: Phase name to latency in ms.

        Returns:
            Dict with mean, median, p95, min, and max.
        """
        if not latencies:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        values = sorted(latencies.values())
        n = len(values)
        p95_idx = min(int(n * 0.95), n - 1)

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": values[p95_idx],
            "min": values[0],
            "max": values[-1],
        }

    def compute_expert_performance(
        self, expert_name: str, feedback_list: List[FeedbackData],
    ) -> Dict[str, Any]:
        """Compute accuracy, latency, and confidence for one expert.

        Args:
            expert_name: The expert to evaluate.
            feedback_list: Feedback entries to filter.

        Returns:
            Dict with accuracy, average_latency_ms, average_confidence.
        """
        relevant = [
            fb for fb in feedback_list if fb.expert_used == expert_name]
        if not relevant:
            return {
                "accuracy": 0.0,
                "average_latency_ms": 0.0,
                "average_confidence": 0.0,
            }

        with_gt = [
            fb for fb in relevant if fb.prediction_correct is not None]
        accuracy = 0.0
        if with_gt:
            accuracy = sum(
                1 for fb in with_gt if fb.prediction_correct) / len(with_gt)

        avg_latency = statistics.mean(fb.total_latency_ms for fb in relevant)
        avg_confidence = statistics.mean(
            fb.expert_confidence for fb in relevant)

        return {
            "accuracy": accuracy,
            "average_latency_ms": avg_latency,
            "average_confidence": avg_confidence,
        }


class FeedbackAggregator:
    """Aggregate multiple :class:`FeedbackData` entries into a summary."""

    def __init__(self) -> None:
        """Initialise the aggregator."""

    def aggregate_feedback(
        self,
        feedback_list: List[FeedbackData],
        time_window: Optional[str] = None,
    ) -> AggregatedFeedback:
        """Aggregate a list of feedback entries.

        Args:
            feedback_list: Individual feedback records.
            time_window: Description of time span; defaults to "all".

        Returns:
            An :class:`AggregatedFeedback` summary.
        """
        start = time.perf_counter()

        num_samples = len(feedback_list)

        # Accuracy
        with_gt = [
            fb for fb in feedback_list
            if fb.prediction_correct is not None]
        overall_accuracy = 0.0
        if with_gt:
            overall_accuracy = sum(
                1 for fb in with_gt if fb.prediction_correct) / len(with_gt)

        # Latency
        average_latency = 0.0
        if feedback_list:
            average_latency = statistics.mean(
                fb.total_latency_ms for fb in feedback_list)

        # User satisfaction
        rated = [fb for fb in feedback_list if fb.user_rating is not None]
        user_satisfaction = 0.0
        if rated:
            ratings = [float(fb.user_rating) for fb in rated
                       if fb.user_rating is not None]
            user_satisfaction = statistics.mean(ratings)

        # Expert performance
        expert_groups: Dict[str, List[FeedbackData]] = {}
        for fb in feedback_list:
            if fb.expert_used:
                expert_groups.setdefault(fb.expert_used, []).append(fb)

        metrics_compute = PerformanceMetricsCompute()
        expert_performance: Dict[str, Dict[str, Any]] = {}
        for name, group in expert_groups.items():
            expert_performance[name] = (
                metrics_compute.compute_expert_performance(name, group))

        elapsed = (time.perf_counter() - start) * 1000.0
        logger.debug("Aggregated %d samples in %.2f ms", num_samples, elapsed)

        return AggregatedFeedback(
            time_period=time_window or "all",
            num_samples=num_samples,
            overall_accuracy=overall_accuracy,
            average_latency_ms=average_latency,
            user_satisfaction=user_satisfaction,
            expert_performance=expert_performance,
        )

    def compute_trend(self, metric_history: List[float]) -> str:
        """Detect whether a metric is improving, declining, or stable.

        Compares the mean of the last third to the first third.
        A change exceeding 5% is flagged.

        Args:
            metric_history: Chronologically ordered metric values.

        Returns:
            One of "improving", "declining", or "stable".
        """
        if len(metric_history) < 3:
            return "stable"

        third = max(len(metric_history) // 3, 1)
        first_mean = statistics.mean(metric_history[:third])
        last_mean = statistics.mean(metric_history[-third:])

        if first_mean == 0.0:
            return "stable"

        change = (last_mean - first_mean) / abs(first_mean)
        if change > 0.05:
            return "improving"
        if change < -0.05:
            return "declining"
        return "stable"

    def identify_failure_modes(
        self, feedback_list: List[FeedbackData],
    ) -> Dict[str, Any]:
        """Group failures by expert and compute failure statistics.

        Args:
            feedback_list: Feedback records to analyse.

        Returns:
            Dict with per-expert failure counts, total failures, and
            failure rate.
        """
        failures = [
            fb for fb in feedback_list if fb.prediction_correct is False]
        total_failures = len(failures)

        expert_failure_counts: Dict[str, int] = {}
        for fb in failures:
            key = fb.expert_used or "unknown"
            expert_failure_counts[key] = expert_failure_counts.get(key, 0) + 1

        with_gt = [
            fb for fb in feedback_list
            if fb.prediction_correct is not None]
        failure_rate = 0.0
        if with_gt:
            failure_rate = total_failures / len(with_gt)

        return {
            "expert_failure_counts": expert_failure_counts,
            "total_failures": total_failures,
            "failure_rate": failure_rate,
        }


class FeedbackCollectionPipeline:
    """Orchestrator for the full Phase 4.1 feedback-collection flow."""

    def __init__(self, config: Optional[Phase3Config] = None) -> None:
        """Initialise the pipeline and its components.

        Args:
            config: Optional Phase-3 configuration.
        """
        self._config = config or Phase3Config()
        self.collector = FeedbackCollector(config=self._config)
        self.metrics_compute = PerformanceMetricsCompute()
        self.aggregator = FeedbackAggregator()
        self._last_metadata: Dict[str, Any] = {}

    def collect(self, execution_results: Dict[str, Any]) -> FeedbackData:
        """Collect feedback from a single execution result dict.

        Args:
            execution_results: Must contain at least input_text and
                prediction. May include ground_truth, user_rating,
                and execution_metrics.

        Returns:
            A :class:`FeedbackData` entry.

        Raises:
            FeedbackCollectionError: When collection fails.
        """
        start = time.perf_counter()

        feedback = self.collector.collect_feedback(
            input_text=execution_results.get("input_text", ""),
            prediction=execution_results.get("prediction"),
            ground_truth=execution_results.get("ground_truth"),
            user_rating=execution_results.get("user_rating"),
            execution_metrics=execution_results.get("execution_metrics"),
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._last_metadata = {
            "feedback_id": feedback.feedback_id,
            "session_id": feedback.session_id,
            "collection_time_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Phase 4.1 collected feedback %s in %.2f ms",
            feedback.feedback_id, elapsed_ms,
        )
        return feedback

    def aggregate(
        self, feedback_history: List[FeedbackData],
    ) -> AggregatedFeedback:
        """Aggregate feedback entries into a summary."""
        return self.aggregator.aggregate_feedback(feedback_history)

    def get_collection_metadata(self) -> Dict[str, Any]:
        """Return a copy of the last collection metadata."""
        return copy.copy(self._last_metadata)
