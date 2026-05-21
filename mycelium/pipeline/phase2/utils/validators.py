"""
Validation utilities for the Phase 2 pipeline.

Provides input validation and result validation helpers used by the
normalization pipeline and downstream phases.

Example:
    >>> from mycelium.pipeline.phase2.utils.validators import InputValidator
    >>> v = InputValidator()
    >>> v.validate_string_type("hello")
    True
    >>> v.validate_not_empty("hello")
    True
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates raw input before it enters the pipeline.

    All methods return ``True`` when validation passes and ``False``
    otherwise.
    """

    def validate_string_type(self, value: Any) -> bool:
        """Check that the input is a string.

        Args:
            value: Value to check.

        Returns:
            ``True`` if *value* is a ``str``.
        """
        return isinstance(value, str)

    def validate_encoding(self, value: str) -> bool:
        """Check that the string is valid UTF-8.

        Args:
            value: String to check.

        Returns:
            ``True`` if *value* can be encoded as UTF-8.
        """
        try:
            if not isinstance(value, str):
                return False
            value.encode("utf-8")
            return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False

    def validate_not_empty(self, value: str) -> bool:
        """Check that the string is not empty after stripping.

        Args:
            value: String to check.

        Returns:
            ``True`` if the stripped string is non-empty.
        """
        if not isinstance(value, str):
            return False
        return len(value.strip()) > 0

    def validate_length_range(
        self,
        value: str,
        min_len: int = 10,
        max_len: int = 50000,
    ) -> bool:
        """Check that the string length falls within a range.

        Args:
            value: String to check.
            min_len: Minimum length (inclusive).
            max_len: Maximum length (inclusive).

        Returns:
            ``True`` if ``min_len <= len(value) <= max_len``.
        """
        if not isinstance(value, str):
            return False
        length = len(value.strip())
        return min_len <= length <= max_len

    def validate_token_range(
        self,
        value: str,
        min_tokens: int = 3,
        max_tokens: int = 10000,
    ) -> bool:
        """Check that the token count falls within a range.

        Tokens are defined as whitespace-separated words.

        Args:
            value: String to check.
            min_tokens: Minimum token count (inclusive).
            max_tokens: Maximum token count (inclusive).

        Returns:
            ``True`` if the token count is within the specified
            range.
        """
        if not isinstance(value, str):
            return False
        token_count = len(value.split())
        return min_tokens <= token_count <= max_tokens


class ResultValidator:
    """Validates outputs produced by the normalization pipeline."""

    def validate_normalization_result(self, result: Any) -> bool:
        """Check that a NormalizationResult has all required fields.

        Args:
            result: Object to validate (expected to be a
                ``NormalizationResult`` dataclass instance).

        Returns:
            ``True`` if all required attributes are present and
            have sensible types.
        """
        required_attrs = [
            "cleaned_text",
            "original_text",
            "language",
            "language_confidence",
            "is_valid",
            "extracted_tags",
            "tag_confidence_scores",
            "domain_mapping",
            "should_reject",
            "rejection_reason",
            "rejection_score",
            "processing_time_ms",
            "validation_report",
            "warnings",
        ]
        for attr in required_attrs:
            if not hasattr(result, attr):
                logger.warning(
                    "NormalizationResult missing attribute: %s",
                    attr,
                )
                return False

        if not isinstance(result.cleaned_text, str):
            return False
        if not isinstance(result.is_valid, bool):
            return False
        if not isinstance(result.extracted_tags, list):
            return False
        if not isinstance(result.warnings, list):
            return False
        return True

    def validate_tag_extraction(self, tags: Any) -> bool:
        """Check that extracted tags are a list of strings.

        Args:
            tags: Value to validate.

        Returns:
            ``True`` if *tags* is a list where every element is a
            string.
        """
        if not isinstance(tags, list):
            return False
        return all(isinstance(t, str) for t in tags)

    def validate_domain_mapping(
        self, mapping: Any
    ) -> bool:
        """Check that a domain mapping has the expected shape.

        Expected shape: ``Dict[str, List[str]]``.

        Args:
            mapping: Value to validate.

        Returns:
            ``True`` if *mapping* matches the expected shape.
        """
        if not isinstance(mapping, dict):
            return False
        for key, value in mapping.items():
            if not isinstance(key, str):
                return False
            if not isinstance(value, list):
                return False
            if not all(isinstance(v, str) for v in value):
                return False
        return True
