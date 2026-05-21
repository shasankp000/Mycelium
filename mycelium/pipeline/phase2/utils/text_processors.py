"""
Text processing utilities for the Phase 2 validation pipeline.

Provides helpers for HTML cleaning, Unicode normalization,
whitespace standardization, and token counting.

Example:
    >>> from mycelium.pipeline.phase2.utils.text_processors import (
    ...     TextProcessor, TokenCounter,
    ... )
    >>> tp = TextProcessor()
    >>> tp.clean_html("<p>Hello</p>")
    'Hello'
    >>> tc = TokenCounter()
    >>> tc.count_tokens("Hello world")
    2
"""

import html
import logging
import re
import unicodedata
from typing import Dict

import bleach
import ftfy

logger = logging.getLogger(__name__)


class TextProcessor:
    """Collection of text cleaning and standardization utilities.

    All methods are stateless and idempotent — applying them
    multiple times produces the same output.
    """

    _CONTROL_CHAR_RE = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
    )

    _MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

    _FANCY_SINGLE_QUOTES = re.compile(r"[\u2018\u2019\u201a\u201b]")
    _FANCY_DOUBLE_QUOTES = re.compile(r"[\u201c\u201d\u201e\u201f]")

    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities.

        Args:
            text: Input text possibly containing HTML markup.

        Returns:
            Plain text with HTML tags stripped and entities decoded.

        Example:
            >>> TextProcessor().clean_html("<b>hi</b> &amp; bye")
            'hi & bye'
        """
        cleaned = bleach.clean(text, tags=[], strip=True)
        cleaned = html.unescape(cleaned)
        return cleaned

    def remove_unicode_control_chars(self, text: str) -> str:
        """Remove Unicode control characters while keeping newlines.

        Args:
            text: Input text.

        Returns:
            Text with control characters removed (newlines, tabs
            preserved).
        """
        return self._CONTROL_CHAR_RE.sub("", text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/tabs into one and trim.

        Args:
            text: Input text.

        Returns:
            Text with normalized whitespace.

        Example:
            >>> TextProcessor().normalize_whitespace("  a   b  ")
            'a b'
        """
        text = self._MULTI_WHITESPACE_RE.sub(" ", text)
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()

    def standardize_quotes(self, text: str) -> str:
        """Replace curly/fancy quotes with ASCII equivalents.

        Args:
            text: Input text.

        Returns:
            Text with standardized quote characters.
        """
        text = self._FANCY_SINGLE_QUOTES.sub("'", text)
        text = self._FANCY_DOUBLE_QUOTES.sub('"', text)
        return text

    def decode_html_entities(self, text: str) -> str:
        """Decode HTML entities to their Unicode equivalents.

        Args:
            text: Input text possibly containing HTML entities.

        Returns:
            Text with HTML entities decoded.

        Example:
            >>> TextProcessor().decode_html_entities("&lt;div&gt;")
            '<div>'
        """
        return html.unescape(text)

    def fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues using ftfy.

        Args:
            text: Input text with potential encoding problems.

        Returns:
            Text with encoding issues corrected.

        Example:
            >>> TextProcessor().fix_encoding_issues("â€™")
            '\\u2019'
        """
        return ftfy.fix_text(text)

    def full_clean(self, text: str) -> str:
        """Apply all cleaning steps in the correct order.

        The order is: fix encoding → clean HTML → remove control
        characters → standardize quotes → normalize whitespace.

        Args:
            text: Raw input text.

        Returns:
            Fully cleaned text.
        """
        text = self.fix_encoding_issues(text)
        text = self.clean_html(text)
        text = self.remove_unicode_control_chars(text)
        text = self.standardize_quotes(text)
        text = self.normalize_whitespace(text)
        return text


class TokenCounter:
    """Simple whitespace-based token counter.

    This provides a fast, dependency-light token counting mechanism
    suitable for input validation. It does **not** use a model
    tokenizer — use a model-specific tokenizer when precision
    matters.

    Example:
        >>> counter = TokenCounter()
        >>> counter.count_tokens("Hello world")
        2
    """

    _WORD_RE = re.compile(r"\S+")

    def count_tokens(self, text: str) -> int:
        """Count tokens (whitespace-separated words) in text.

        Args:
            text: Input text.

        Returns:
            Number of whitespace-separated tokens.
        """
        return len(self._WORD_RE.findall(text))

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count using a character-based heuristic.

        Uses the rule-of-thumb that one token ≈ 4 characters for
        English text (GPT-style tokenizers).

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // 4) if text.strip() else 0

    def get_token_distribution(self, text: str) -> Dict[str, int]:
        """Get distribution statistics about tokens.

        Args:
            text: Input text.

        Returns:
            Dictionary with ``total_tokens``, ``unique_tokens``,
            ``avg_token_length``, and ``max_token_length``.
        """
        tokens = self._WORD_RE.findall(text)
        if not tokens:
            return {
                "total_tokens": 0,
                "unique_tokens": 0,
                "avg_token_length": 0,
                "max_token_length": 0,
            }
        lengths = [len(t) for t in tokens]
        return {
            "total_tokens": len(tokens),
            "unique_tokens": len(set(tokens)),
            "avg_token_length": sum(lengths) // len(lengths),
            "max_token_length": max(lengths),
        }
