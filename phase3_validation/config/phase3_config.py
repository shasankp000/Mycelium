"""
Phase 3-5 Configuration Module for the Mycelium Expert System.

Provides configuration for action execution, feedback collection,
performance analysis, and continuous improvement phases.

Example:
    >>> from phase3_validation.config.phase3_config import Phase3Config
    >>> config = Phase3Config()
    >>> print(config.action_timeout_seconds)
    30.0
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    """Read an integer from an environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(
            f"Environment variable {key} must be an integer, "
            f"got: {val!r}"
        )


def _env_float(key: str, default: float) -> float:
    """Read a float from an environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(
            f"Environment variable {key} must be a float, "
            f"got: {val!r}"
        )


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean from an environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_csv_list(key: str, default: List[str]) -> List[str]:
    """Read a comma-separated list from environment."""
    val = os.environ.get(key)
    if val is None:
        return list(default)
    parsed = [item.strip() for item in val.split(",") if item.strip()]
    return parsed or list(default)


@dataclass
class Phase3Config:
    """Configuration for the Phase 3-5 pipeline.

    All parameters have sensible defaults and can be overridden
    via environment variables prefixed with ``P3_``.

    Attributes:
        action_timeout_seconds: Max seconds for action execution.
        enable_rollback: Whether rollback is enabled on failure.
        max_created_resources: Max resources per execution.
        feedback_window_hours: Default feedback aggregation window.
        min_samples_for_analysis: Minimum samples for analysis.
        accuracy_threshold: Minimum acceptable accuracy.
        latency_threshold_ms: Maximum acceptable latency.
        health_score_threshold: Minimum system health score.
        improvement_cycle_interval_hours: Hours between cycles.
        max_parameter_change_pct: Max parameter change percent.
        enable_auto_retraining: Whether auto-retraining enabled.
        min_retraining_samples: Minimum samples for retraining.
        log_level: Logging level string.
        enable_performance_logging: Whether to log timings.
    """

    # --- Action execution ---
    action_timeout_seconds: float = field(
        default_factory=lambda: _env_float(
            "P3_ACTION_TIMEOUT", 30.0
        )
    )
    enable_rollback: bool = field(
        default_factory=lambda: _env_bool(
            "P3_ENABLE_ROLLBACK", True
        )
    )
    max_created_resources: int = field(
        default_factory=lambda: _env_int(
            "P3_MAX_RESOURCES", 10
        )
    )

    # --- Feedback collection ---
    feedback_window_hours: int = field(
        default_factory=lambda: _env_int(
            "P3_FEEDBACK_WINDOW_HOURS", 24
        )
    )
    min_samples_for_analysis: int = field(
        default_factory=lambda: _env_int(
            "P3_MIN_SAMPLES", 5
        )
    )

    # --- Performance thresholds ---
    accuracy_threshold: float = field(
        default_factory=lambda: _env_float(
            "P3_ACCURACY_THRESHOLD", 0.7
        )
    )
    latency_threshold_ms: float = field(
        default_factory=lambda: _env_float(
            "P3_LATENCY_THRESHOLD_MS", 5000.0
        )
    )
    health_score_threshold: float = field(
        default_factory=lambda: _env_float(
            "P3_HEALTH_THRESHOLD", 60.0
        )
    )

    # --- Improvement loop ---
    improvement_cycle_interval_hours: int = field(
        default_factory=lambda: _env_int(
            "P3_IMPROVEMENT_INTERVAL", 24
        )
    )
    max_parameter_change_pct: float = field(
        default_factory=lambda: _env_float(
            "P3_MAX_PARAM_CHANGE", 20.0
        )
    )
    enable_auto_retraining: bool = field(
        default_factory=lambda: _env_bool(
            "P3_AUTO_RETRAIN", False
        )
    )
    min_retraining_samples: int = field(
        default_factory=lambda: _env_int(
            "P3_MIN_RETRAIN_SAMPLES", 50
        )
    )

    # --- Logging ---
    log_level: str = "INFO"
    enable_performance_logging: bool = field(
        default_factory=lambda: _env_bool(
            "P3_ENABLE_PERF_LOGGING", True
        )
    )

    # --- Validation gate behavior ---
    enforce_validation_gate: bool = field(
        default_factory=lambda: _env_bool(
            "P3_ENFORCE_VALIDATION_GATE", True
        )
    )
    action_allowed_result_classes: List[str] = field(
        default_factory=lambda: _env_csv_list(
            "P3_ACTION_ALLOWED_RESULT_CLASSES",
            ["all_pass"],
        )
    )

    def validate(self) -> List[str]:
        """Validate all configuration values.

        Returns:
            A list of validation error messages. Empty means valid.
        """
        errors: List[str] = []

        if self.action_timeout_seconds <= 0:
            errors.append(
                "action_timeout_seconds must be > 0"
            )
        if self.max_created_resources < 1:
            errors.append(
                "max_created_resources must be >= 1"
            )
        if not (0.0 <= self.accuracy_threshold <= 1.0):
            errors.append(
                "accuracy_threshold must be in [0, 1]"
            )
        if self.latency_threshold_ms <= 0:
            errors.append(
                "latency_threshold_ms must be > 0"
            )
        if not (0.0 <= self.health_score_threshold <= 100.0):
            errors.append(
                "health_score_threshold must be in [0, 100]"
            )
        if not (0.0 <= self.max_parameter_change_pct <= 100.0):
            errors.append(
                "max_parameter_change_pct must be in [0, 100]"
            )
        if self.min_retraining_samples < 1:
            errors.append(
                "min_retraining_samples must be >= 1"
            )
        if not self.action_allowed_result_classes:
            errors.append(
                "action_allowed_result_classes must contain at least one class"
            )

        if errors:
            logger.warning(
                "Configuration validation errors: %s", errors
            )

        return errors
