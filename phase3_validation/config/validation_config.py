"""
Validation Configuration for Phase 3 Validation Layer.

Defines thresholds, parameters, and settings used by the
contradiction analyzer and validation orchestrator.

Example:
    >>> from phase3_validation.config.validation_config import (
    ...     ValidationConfig,
    ... )
    >>> cfg = ValidationConfig()
    >>> print(cfg.contradiction_threshold)
    0.65
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


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


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean from an environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


@dataclass
class ValidationConfig:
    """Configuration for the Phase 3 Validation Layer.

    All parameters have sensible defaults and can be overridden
    via environment variables prefixed with ``VAL_``.

    Attributes:
        contradiction_threshold: Cosine similarity below which
            an answer is considered contradictory to reasoning.
            Default 0.65.
        ood_threshold: Mahalanobis distance above which an
            answer is considered out-of-distribution (hallucination).
            Default 2.5 (sigma).
        reasoning_chain_threshold: Minimum cosine similarity
            between consecutive reasoning steps.
            Default 0.5.
        evidence_alignment_threshold: Fraction of answer concepts
            that must appear in evidence (0 to 1).
            Default 0.5.
        max_retries: Maximum number of pipeline re-runs allowed
            before escalating. Default 3.
        embedding_dim: Dimension for simple embeddings.
            Default 128.
        use_simple_embeddings: Whether to use the lightweight
            embedding backend instead of sentence-transformers.
            Default True (avoids model download overhead).
        sentence_transformer_model: Model name to use when
            use_simple_embeddings is False.
    """

    # Contradiction detection
    contradiction_threshold: float = field(
        default_factory=lambda: _env_float("VAL_CONTRADICTION_THRESHOLD", 0.65)
    )

    # OOD / hallucination detection
    ood_threshold: float = field(
        default_factory=lambda: _env_float("VAL_OOD_THRESHOLD", 2.5)
    )

    # Reasoning chain validity
    reasoning_chain_threshold: float = field(
        default_factory=lambda: _env_float("VAL_CHAIN_THRESHOLD", 0.5)
    )

    # Evidence alignment
    evidence_alignment_threshold: float = field(
        default_factory=lambda: _env_float("VAL_EVIDENCE_THRESHOLD", 0.5)
    )

    # Re-run loop prevention
    max_retries: int = field(
        default_factory=lambda: _env_int("VAL_MAX_RETRIES", 3)
    )

    # Embedding settings
    embedding_dim: int = field(
        default_factory=lambda: _env_int("VAL_EMBEDDING_DIM", 128)
    )
    use_simple_embeddings: bool = field(
        default_factory=lambda: _env_bool("VAL_USE_SIMPLE_EMBEDDINGS", True)
    )
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    # Severity mapping
    severity_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "CRITICAL": 1.0,
            "MAJOR": 0.7,
            "MINOR": 0.3,
            "NONE": 0.0,
        }
    )

    def validate(self) -> list:
        """Validate configuration values.

        Returns:
            A list of validation error messages. Empty means valid.
        """
        errors = []
        if not (0.0 < self.contradiction_threshold < 1.0):
            errors.append(
                "contradiction_threshold must be in (0, 1)"
            )
        if self.ood_threshold <= 0.0:
            errors.append("ood_threshold must be > 0")
        if not (0.0 < self.reasoning_chain_threshold < 1.0):
            errors.append(
                "reasoning_chain_threshold must be in (0, 1)"
            )
        if not (0.0 < self.evidence_alignment_threshold <= 1.0):
            errors.append(
                "evidence_alignment_threshold must be in (0, 1]"
            )
        if self.max_retries < 1:
            errors.append("max_retries must be >= 1")
        if self.embedding_dim < 8:
            errors.append("embedding_dim must be >= 8")
        if errors:
            logger.warning(
                "ValidationConfig errors: %s", errors
            )
        return errors
