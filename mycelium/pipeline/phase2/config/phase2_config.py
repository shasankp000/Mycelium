"""
Phase 2 Configuration Module for the Mycelium Expert System.

Provides comprehensive configuration for all phases of the reasoning
pipeline, including input validation thresholds, language detection,
domain tag extraction parameters, and pipeline performance settings.

All configuration values support environment variable overrides and
include validation logic.

Example:
    >>> from mycelium.pipeline.phase2.config.phase2_config import Phase2Config
    >>> config = Phase2Config()
    >>> print(config.input_min_length)
    10
    >>> print(config.similarity_threshold)
    0.45
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    """Read an integer from an environment variable with a default.

    Args:
        key: Environment variable name.
        default: Default value when the variable is unset.

    Returns:
        The parsed integer value.

    Raises:
        ValueError: If the environment variable value is not a valid
            integer.
    """
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
    """Read a float from an environment variable with a default.

    Args:
        key: Environment variable name.
        default: Default value when the variable is unset.

    Returns:
        The parsed float value.

    Raises:
        ValueError: If the environment variable value is not a valid
            float.
    """
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


def _env_str(key: str, default: str) -> str:
    """Read a string from an environment variable with a default.

    Args:
        key: Environment variable name.
        default: Default value when the variable is unset.

    Returns:
        The string value.
    """
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean from an environment variable with a default.

    Args:
        key: Environment variable name.
        default: Default value when the variable is unset.

    Returns:
        The parsed boolean value.
    """
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


@dataclass
class Phase2Config:
    """Configuration for the Phase 2 validation pipeline.

    All parameters have sensible defaults and can be overridden via
    environment variables prefixed with ``P2_``.

    Attributes:
        input_min_length: Minimum character length for valid input.
        input_max_length: Maximum character length for valid input.
        input_min_tokens: Minimum token count for valid input.
        input_max_tokens: Maximum token count for valid input.
        supported_languages: List of supported ISO-639-1 language codes.
        language_confidence_threshold: Minimum confidence for language
            detection to be considered reliable.
        similarity_threshold: Cosine similarity threshold for domain
            tag extraction (from auto_semantic_clusterer).
        tag_confidence_threshold: Minimum confidence for a tag to be
            kept after filtering.
        max_tags: Maximum number of tags to extract per input.
        spam_score_threshold: Score above which input is considered
            spam.
        rejection_score_threshold: Combined rejection score threshold.
        coherence_min_score: Minimum coherence score for valid input.
        log_level: Logging level string.
        enable_performance_logging: Whether to log timing information.
        pipeline_timeout_seconds: Maximum seconds for the full
            pipeline.
        use_auto_clustering: Whether to use automatic semantic
            clustering for domain tag extraction.
        embedding_model: Sentence transformer model name.
    """

    # --- Input validation thresholds ---
    input_min_length: int = field(
        default_factory=lambda: _env_int("P2_INPUT_MIN_LENGTH", 10)
    )
    input_max_length: int = field(
        default_factory=lambda: _env_int("P2_INPUT_MAX_LENGTH", 50000)
    )
    input_min_tokens: int = field(
        default_factory=lambda: _env_int("P2_INPUT_MIN_TOKENS", 3)
    )
    input_max_tokens: int = field(
        default_factory=lambda: _env_int("P2_INPUT_MAX_TOKENS", 10000)
    )

    # --- Language detection ---
    supported_languages: List[str] = field(
        default_factory=lambda: ["en"]
    )
    language_confidence_threshold: float = field(
        default_factory=lambda: _env_float(
            "P2_LANG_CONFIDENCE_THRESHOLD", 0.5
        )
    )

    # --- Domain tag extraction ---
    similarity_threshold: float = field(
        default_factory=lambda: _env_float(
            "P2_SIMILARITY_THRESHOLD", 0.45
        )
    )
    tag_confidence_threshold: float = field(
        default_factory=lambda: _env_float(
            "P2_TAG_CONFIDENCE_THRESHOLD", 0.45
        )
    )
    max_tags: int = field(
        default_factory=lambda: _env_int("P2_MAX_TAGS", 10)
    )

    # --- Early rejection rules ---
    spam_score_threshold: float = field(
        default_factory=lambda: _env_float(
            "P2_SPAM_SCORE_THRESHOLD", 0.7
        )
    )
    rejection_score_threshold: float = field(
        default_factory=lambda: _env_float(
            "P2_REJECTION_SCORE_THRESHOLD", 0.6
        )
    )
    coherence_min_score: float = field(
        default_factory=lambda: _env_float(
            "P2_COHERENCE_MIN_SCORE", 0.3
        )
    )

    # --- Logging ---
    log_level: str = field(
        default_factory=lambda: _env_str("P2_LOG_LEVEL", "INFO")
    )
    enable_performance_logging: bool = field(
        default_factory=lambda: _env_bool(
            "P2_ENABLE_PERF_LOGGING", True
        )
    )

    # --- Pipeline performance ---
    pipeline_timeout_seconds: float = field(
        default_factory=lambda: _env_float(
            "P2_PIPELINE_TIMEOUT", 30.0
        )
    )

    # --- Integration ---
    use_auto_clustering: bool = field(
        default_factory=lambda: _env_bool(
            "P2_USE_AUTO_CLUSTERING", True
        )
    )
    embedding_model: str = field(
        default_factory=lambda: _env_str(
            "P2_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
    )

    def validate(self) -> List[str]:
        """Validate all configuration values.

        Returns:
            A list of validation error messages. An empty list
            indicates all values are valid.

        Example:
            >>> config = Phase2Config()
            >>> errors = config.validate()
            >>> assert len(errors) == 0
        """
        errors: List[str] = []

        if self.input_min_length < 0:
            errors.append("input_min_length must be >= 0")
        if self.input_max_length <= self.input_min_length:
            errors.append(
                "input_max_length must be > input_min_length"
            )
        if self.input_min_tokens < 0:
            errors.append("input_min_tokens must be >= 0")
        if self.input_max_tokens <= self.input_min_tokens:
            errors.append(
                "input_max_tokens must be > input_min_tokens"
            )

        if not (0.0 <= self.language_confidence_threshold <= 1.0):
            errors.append(
                "language_confidence_threshold must be in [0, 1]"
            )
        if not (0.0 <= self.similarity_threshold <= 1.0):
            errors.append(
                "similarity_threshold must be in [0, 1]"
            )
        if not (0.0 <= self.tag_confidence_threshold <= 1.0):
            errors.append(
                "tag_confidence_threshold must be in [0, 1]"
            )
        if not (0.0 <= self.spam_score_threshold <= 1.0):
            errors.append(
                "spam_score_threshold must be in [0, 1]"
            )
        if not (0.0 <= self.rejection_score_threshold <= 1.0):
            errors.append(
                "rejection_score_threshold must be in [0, 1]"
            )
        if not (0.0 <= self.coherence_min_score <= 1.0):
            errors.append(
                "coherence_min_score must be in [0, 1]"
            )

        if self.max_tags < 1:
            errors.append("max_tags must be >= 1")
        if self.pipeline_timeout_seconds <= 0:
            errors.append(
                "pipeline_timeout_seconds must be > 0"
            )

        if not self.supported_languages:
            errors.append(
                "supported_languages must not be empty"
            )

        valid_levels = {
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
        }
        if self.log_level.upper() not in valid_levels:
            errors.append(
                f"log_level must be one of {valid_levels}"
            )

        if errors:
            logger.warning(
                "Configuration validation errors: %s", errors
            )

        return errors

    @classmethod
    def from_yaml(cls, path: str) -> "Phase2Config":
        """Load configuration from a YAML file.

        Args:
            path: Filesystem path to the YAML configuration file.

        Returns:
            A Phase2Config instance populated from the file.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the specified file does not exist.

        Example:
            >>> config = Phase2Config.from_yaml("config.yaml")
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML configuration. "
                "Install it with: pip install pyyaml"
            )

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(
                    "Unknown config key in YAML: %s", key
                )
        return config


# Module-level singleton for convenience.
PHASE2_CONFIG = Phase2Config()
