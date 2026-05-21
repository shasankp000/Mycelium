"""Phase 4.2 – Performance Analyzer.

Analyzes system and expert performance using aggregated feedback from
Phase 4.1, identifies issues, computes severity scores, and generates
improvement recommendations for the optimisation loop.

Example::

    pipeline = PerformanceAnalysisPipeline()
    analysis = pipeline.analyze(aggregated_feedback)
    if analysis.action_required:
        print(analysis.improvement_recommendations)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mycelium.pipeline.phase3.config.phase3_config import Phase3Config
from mycelium.pipeline.phase3.utils.types import (
    AggregatedFeedback,
    ExpertAnalysis,
    PerformanceAnalysis,
)

logger = logging.getLogger(__name__)

_ISSUE_BASE_SEVERITY: Dict[str, float] = {
    "low_accuracy": 0.8,
    "high_latency": 0.5,
    "low_user_satisfaction": 0.7,
    "high_calibration_error": 0.6,
    "declining_trend": 0.4,
}

_RECOMMENDATION_MAP: Dict[str, str] = {
    "low_accuracy": (
        "Retrain experts to improve accuracy – current accuracy"
        " is below the configured threshold."
    ),
    "high_latency": (
        "Optimize pipeline latency – average response time"
        " exceeds the acceptable limit."
    ),
    "low_user_satisfaction": (
        "Investigate user experience – satisfaction score"
        " is critically low."
    ),
    "high_calibration_error": (
        "Recalibrate confidence outputs – calibration error"
        " is above the acceptable range."
    ),
    "declining_trend": (
        "Review recent changes – one or more metrics show"
        " a declining trend."
    ),
}


class PerformanceAnalysisError(Exception):
    """Raised when performance analysis fails."""


class PerformanceAnalyzer:
    """Analyze overall system performance and identify issues."""

    def __init__(self, config: Optional[Phase3Config] = None) -> None:
        """Initialise the analyser.

        Args:
            config: Optional configuration; defaults when *None*.
        """
        self._config = config or Phase3Config()
        self._analysis_history: List[PerformanceAnalysis] = []

    def analyze_performance(
        self,
        aggregated_feedback: AggregatedFeedback,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> PerformanceAnalysis:
        """Run a full performance analysis cycle.

        Args:
            aggregated_feedback: Metrics produced by Phase 4.1.
            thresholds: Optional overrides (``accuracy_threshold``,
                ``latency_threshold_ms``).

        Returns:
            A populated :class:`PerformanceAnalysis`.

        Raises:
            PerformanceAnalysisError: On unexpected failure.
        """
        start = time.perf_counter()
        try:
            eff = self._resolve_thresholds(thresholds)
            logger.info(
                "Starting performance analysis (period=%s, n=%d)",
                aggregated_feedback.time_period,
                aggregated_feedback.num_samples,
            )

            issues = self.identify_performance_issues(
                aggregated_feedback, eff,
            )
            severity_scores: Dict[str, float] = {
                iss["type"]: self.compute_issue_severity(iss, eff)
                for iss in issues
            }
            recommendations = self.generate_improvement_recommendations(
                issues,
            )
            health = self._compute_health_score(aggregated_feedback)
            action_required = (
                health < self._config.health_score_threshold
            )

            now = datetime.now(timezone.utc).isoformat()
            analysis = PerformanceAnalysis(
                analysis_timestamp=now,
                time_period=aggregated_feedback.time_period,
                overall_accuracy=aggregated_feedback.overall_accuracy,
                overall_latency_ms=aggregated_feedback.average_latency_ms,
                identified_issues=issues,
                issue_severity_scores=severity_scores,
                improvement_recommendations=recommendations,
                system_health_score=health,
                action_required=action_required,
            )
            self._analysis_history.append(analysis)

            elapsed = (time.perf_counter() - start) * 1000.0
            logger.info(
                "Analysis done in %.2f ms (health=%.1f, act=%s)",
                elapsed, health, action_required,
            )
            return analysis
        except PerformanceAnalysisError:
            raise
        except Exception as exc:
            raise PerformanceAnalysisError(
                f"Performance analysis failed: {exc}"
            ) from exc

    def identify_performance_issues(
        self,
        aggregated_feedback: AggregatedFeedback,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Identify performance issues from aggregated metrics.

        Args:
            aggregated_feedback: Metrics from Phase 4.1.
            thresholds: Resolved threshold dict (optional).

        Returns:
            List of issue dicts with *type*, *severity*,
            *description*, *metric_value*, and *threshold*.
        """
        eff = self._resolve_thresholds(thresholds)
        acc_t = eff["accuracy_threshold"]
        lat_t = eff["latency_threshold_ms"]
        fb = aggregated_feedback
        issues: List[Dict[str, Any]] = []

        if fb.overall_accuracy < acc_t:
            issues.append({
                "type": "low_accuracy",
                "severity": "high",
                "description": (
                    f"Overall accuracy {fb.overall_accuracy:.3f}"
                    f" is below threshold {acc_t:.3f}."
                ),
                "metric_value": fb.overall_accuracy,
                "threshold": acc_t,
            })
        if fb.average_latency_ms > lat_t:
            issues.append({
                "type": "high_latency",
                "severity": "medium",
                "description": (
                    f"Average latency {fb.average_latency_ms:.1f} ms"
                    f" exceeds threshold {lat_t:.1f} ms."
                ),
                "metric_value": fb.average_latency_ms,
                "threshold": lat_t,
            })
        if fb.user_satisfaction < 3.0:
            issues.append({
                "type": "low_user_satisfaction",
                "severity": "high",
                "description": (
                    f"User satisfaction {fb.user_satisfaction:.2f}"
                    " is below 3.0."
                ),
                "metric_value": fb.user_satisfaction,
                "threshold": 3.0,
            })
        if fb.overall_calibration_error > 0.15:
            issues.append({
                "type": "high_calibration_error",
                "severity": "medium",
                "description": (
                    f"Calibration error"
                    f" {fb.overall_calibration_error:.3f}"
                    " exceeds 0.150."
                ),
                "metric_value": fb.overall_calibration_error,
                "threshold": 0.15,
            })
        for metric, trend in fb.trends.items():
            if trend == "declining":
                issues.append({
                    "type": "declining_trend",
                    "severity": "low",
                    "description": (
                        f"Metric '{metric}' shows a declining"
                        " trend."
                    ),
                    "metric_value": 0.0,
                    "threshold": 0.0,
                })

        logger.debug("Identified %d performance issues", len(issues))
        return issues

    def compute_issue_severity(
        self,
        issue: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute a normalised severity score for *issue*.

        Args:
            issue: Issue dict from :meth:`identify_performance_issues`.
            thresholds: Resolved threshold dict (optional).

        Returns:
            Severity in ``[0, 1]``.
        """
        itype: str = issue.get("type", "")
        base = _ISSUE_BASE_SEVERITY.get(itype, 0.5)
        val: float = issue.get("metric_value", 0.0)
        thr: float = issue.get("threshold", 0.0)

        if thr == 0.0:
            return min(max(base, 0.0), 1.0)

        if itype in ("high_latency", "high_calibration_error"):
            deviation = (val - thr) / thr if thr > 0 else 0.0
        else:
            deviation = (thr - val) / thr if thr > 0 else 0.0

        scaled = base + 0.2 * min(max(deviation, 0.0), 1.0)
        return min(max(scaled, 0.0), 1.0)

    def generate_improvement_recommendations(
        self, issues: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate actionable improvement recommendations.

        Args:
            issues: Issue list from :meth:`identify_performance_issues`.

        Returns:
            Deduplicated list of recommendation strings.
        """
        seen: set[str] = set()
        recs: List[str] = []
        for issue in issues:
            itype: str = issue.get("type", "")
            if itype in seen:
                continue
            seen.add(itype)
            rec = _RECOMMENDATION_MAP.get(itype)
            if rec:
                recs.append(rec)
        if recs:
            logger.info("Generated %d recommendations", len(recs))
        return recs

    # ── private helpers ─────────────────────────────────────────

    def _resolve_thresholds(
        self, overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Merge caller overrides with config defaults."""
        defaults: Dict[str, float] = {
            "accuracy_threshold": self._config.accuracy_threshold,
            "latency_threshold_ms": self._config.latency_threshold_ms,
        }
        if overrides:
            defaults.update(overrides)
        return defaults

    def _compute_health_score(
        self, feedback: AggregatedFeedback,
    ) -> float:
        """Compute weighted system health score (0–100).

        Weights: accuracy 0.50, latency 0.30, satisfaction 0.20.
        """
        acc = feedback.overall_accuracy * 100.0
        lat_thresh = self._config.latency_threshold_ms
        if lat_thresh > 0:
            lat = (1.0 - min(
                feedback.average_latency_ms / lat_thresh, 1.0,
            )) * 100.0
        else:
            lat = 100.0
        sat = min(feedback.user_satisfaction / 5.0, 1.0) * 100.0
        health = 0.50 * acc + 0.30 * lat + 0.20 * sat
        return min(max(health, 0.0), 100.0)


class ExpertPerformanceAnalyzer:
    """Analyze individual expert performance."""

    def __init__(self, config: Optional[Phase3Config] = None) -> None:
        """Initialise the expert analyser.

        Args:
            config: Optional configuration; defaults when *None*.
        """
        self._config = config or Phase3Config()
        self._expert_history: Dict[str, List[ExpertAnalysis]] = {}

    def analyze_expert(
        self, expert_name: str, feedback: AggregatedFeedback,
    ) -> ExpertAnalysis:
        """Analyze a single expert's performance.

        Args:
            expert_name: Key in ``feedback.expert_performance``.
            feedback: Aggregated metrics from Phase 4.1.

        Returns:
            A populated :class:`ExpertAnalysis`.

        Raises:
            PerformanceAnalysisError: If expert not in *feedback*.
        """
        start = time.perf_counter()
        metrics = feedback.expert_performance.get(expert_name)
        if metrics is None:
            raise PerformanceAnalysisError(
                f"Expert '{expert_name}' not found in feedback"
            )

        accuracy = float(metrics.get("accuracy", 0.0))
        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = float(metrics.get("f1_score", 0.0))

        latency = float(metrics.get("average_latency_ms", 0.0))
        cal_err = float(
            metrics.get("confidence_calibration_error", 0.0),
        )
        trend = str(metrics.get("performance_trend", "stable"))
        reliability = self.compute_expert_reliability_score(
            expert_name, feedback,
        )
        total = int(metrics.get("total_samples", 0))

        analysis = ExpertAnalysis(
            expert_name=expert_name,
            total_samples=total,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            average_latency_ms=latency,
            confidence_calibration_error=cal_err,
            reliability_score=reliability,
            performance_trend=trend,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        self._expert_history.setdefault(expert_name, []).append(
            analysis,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        logger.debug(
            "Analyzed expert '%s' in %.2f ms (acc=%.3f, rel=%.3f)",
            expert_name, elapsed, accuracy, reliability,
        )
        return analysis

    def detect_expert_degradation(
        self, expert_name: str, feedback: AggregatedFeedback,
    ) -> bool:
        """Detect whether an expert is degrading.

        Args:
            expert_name: Expert key.
            feedback: Aggregated feedback.

        Returns:
            *True* if trend is ``"declining"`` or accuracy is
            below the configured threshold.
        """
        metrics = feedback.expert_performance.get(expert_name, {})
        trend = str(metrics.get("performance_trend", "stable"))
        accuracy = float(metrics.get("accuracy", 0.0))
        degraded = (
            trend == "declining"
            or accuracy < self._config.accuracy_threshold
        )
        if degraded:
            logger.warning(
                "Expert '%s' degradation (trend=%s, acc=%.3f)",
                expert_name, trend, accuracy,
            )
        return degraded

    def compute_expert_reliability_score(
        self, expert_name: str, feedback: AggregatedFeedback,
    ) -> float:
        """Compute weighted reliability score for an expert.

        Weights: accuracy 0.4, (1-cal_error) 0.3,
        (1-latency_norm) 0.2, user_sat_norm 0.1.

        Returns:
            Reliability clamped to ``[0, 1]``.
        """
        metrics = feedback.expert_performance.get(expert_name, {})
        accuracy = float(metrics.get("accuracy", 0.0))
        cal_err = float(
            metrics.get("confidence_calibration_error", 0.0),
        )
        latency = float(metrics.get("average_latency_ms", 0.0))
        lat_thresh = self._config.latency_threshold_ms
        lat_norm = (
            min(latency / lat_thresh, 1.0) if lat_thresh > 0 else 0.0
        )
        user_sat = float(
            metrics.get("user_satisfaction", feedback.user_satisfaction),
        )
        sat_norm = min(user_sat / 5.0, 1.0)

        score = (
            0.4 * accuracy
            + 0.3 * (1.0 - cal_err)
            + 0.2 * (1.0 - lat_norm)
            + 0.1 * sat_norm
        )
        return min(max(score, 0.0), 1.0)


class PerformanceAnalysisPipeline:
    """Orchestrate system and expert performance analysis."""

    def __init__(self, config: Optional[Phase3Config] = None) -> None:
        """Initialise the pipeline.

        Args:
            config: Optional configuration; defaults when *None*.
        """
        self._config = config or Phase3Config()
        self._analyzer = PerformanceAnalyzer(self._config)
        self._expert_analyzer = ExpertPerformanceAnalyzer(self._config)
        self._last_metadata: Dict[str, Any] = {}

    def analyze(
        self, aggregated_feedback: AggregatedFeedback,
    ) -> PerformanceAnalysis:
        """Run end-to-end performance analysis.

        Performs system-level analysis then analyses every expert
        in ``aggregated_feedback.expert_performance``.

        Args:
            aggregated_feedback: Metrics produced by Phase 4.1.

        Returns:
            :class:`PerformanceAnalysis` with per-expert analyses.

        Raises:
            PerformanceAnalysisError: On any analysis failure.
        """
        start = time.perf_counter()
        logger.info("Performance analysis pipeline started")
        try:
            analysis = self._analyzer.analyze_performance(
                aggregated_feedback,
            )

            expert_analyses: List[ExpertAnalysis] = []
            for name in aggregated_feedback.expert_performance:
                ea = self._expert_analyzer.analyze_expert(
                    name, aggregated_feedback,
                )
                expert_analyses.append(ea)
                if self._expert_analyzer.detect_expert_degradation(
                    name, aggregated_feedback,
                ):
                    logger.warning(
                        "Expert '%s' flagged for degradation", name,
                    )
            analysis.expert_analyses = expert_analyses

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._last_metadata = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round(elapsed_ms, 2),
                "num_experts_analyzed": len(expert_analyses),
                "num_issues": len(analysis.identified_issues),
                "health_score": analysis.system_health_score,
                "action_required": analysis.action_required,
            }
            logger.info(
                "Pipeline complete in %.2f ms (%d experts, %d issues)",
                elapsed_ms, len(expert_analyses),
                len(analysis.identified_issues),
            )
            return analysis
        except PerformanceAnalysisError:
            raise
        except Exception as exc:
            raise PerformanceAnalysisError(
                f"Analysis pipeline failed: {exc}"
            ) from exc

    def get_analysis_metadata(self) -> Dict[str, Any]:
        """Return metadata from the last analysis run."""
        return dict(self._last_metadata)
