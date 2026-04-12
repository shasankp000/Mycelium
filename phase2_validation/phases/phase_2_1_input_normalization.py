"""
Phase 2.1 — Input Normalization & Pre-filtering.

This module implements the first phase of the six-phase reasoning
pipeline. It cleans, validates, and enriches raw user input before
it is passed to downstream phases.

Components:
    * TextNormalizer — text cleaning & standardization
    * LanguageValidator — language detection & validation
    * ContentValidator — length, token, coherence checks
    * DomainTagExtractor — tag extraction via semantic clustering
    * EarlyRejectionFilter — spam / gibberish / off-topic filter
    * InputNormalizationPipeline — orchestrator that runs all steps

Example:
    >>> from phase2_validation.phases.phase_2_1_input_normalization import (
    ...     InputNormalizationPipeline,
    ... )
    >>> pipeline = InputNormalizationPipeline()
    >>> result = pipeline.normalize(
    ...     "This is a test sentence about medical treatment."
    ... )
    >>> print(result.is_valid)
    True
"""

import logging
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from phase2_validation.config.phase2_config import Phase2Config
from phase2_validation.utils.text_processors import (
    TextProcessor,
    TokenCounter,
)
from phase2_validation.utils.validators import (
    InputValidator,
    ResultValidator,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class InputNormalizationError(Exception):
    """Base exception for input normalization failures."""


class InvalidInputTypeError(InputNormalizationError):
    """Raised when the input is not a string."""


class InputTooShortError(InputNormalizationError):
    """Raised when the input is shorter than the minimum length."""


class InputTooLongError(InputNormalizationError):
    """Raised when the input exceeds the maximum length."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """Structured result returned by the normalization pipeline.

    Attributes:
        cleaned_text: The text after all cleaning steps.
        original_text: The raw input text.
        language: Detected ISO-639-1 language code.
        language_confidence: Confidence of the language detection.
        is_valid: Whether the input passed all validation checks.
        extracted_tags: Domain tags extracted from the text.
        tag_confidence_scores: Confidence score per extracted tag.
        domain_mapping: Mapping of tags to canonical domains.
        should_reject: Whether the input should be rejected.
        rejection_reason: Human-readable rejection reason (if any).
        rejection_score: Numerical rejection score in ``[0, 1]``.
        processing_time_ms: Total wall-clock processing time.
        validation_report: Detailed per-check validation results.
        warnings: Non-fatal issues encountered during processing.
    """

    cleaned_text: str = ""
    original_text: str = ""
    language: str = "en"
    language_confidence: float = 0.0
    is_valid: bool = False
    extracted_tags: List[str] = field(default_factory=list)
    tag_confidence_scores: Dict[str, float] = field(
        default_factory=dict
    )
    domain_mapping: Dict[str, List[str]] = field(
        default_factory=dict
    )
    should_reject: bool = False
    rejection_reason: Optional[str] = None
    rejection_score: float = 0.0
    processing_time_ms: float = 0.0
    validation_report: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sub-components
# ---------------------------------------------------------------------------


class TextNormalizer:
    """Text cleaning and standardization.

    Wraps the lower-level ``TextProcessor`` with additional steps
    such as HTML removal, special-character filtering, whitespace
    normalization, control-character removal, Unicode normalization,
    and encoding standardization.
    """

    _SPECIAL_CHAR_RE = re.compile(
        r"[^\w\s.,!?;:'\"\-\(\)\[\]{}/@#$%&*+=<>]",
        re.UNICODE,
    )

    def __init__(self) -> None:
        self._processor = TextProcessor()

    def remove_html_tags(self, text: str) -> str:
        """Strip HTML tags and decode entities.

        Args:
            text: Input text.

        Returns:
            Plain text.
        """
        return self._processor.clean_html(text)

    def remove_special_characters(
        self, text: str, preserve_punctuation: bool = True
    ) -> str:
        """Remove special characters from text.

        Args:
            text: Input text.
            preserve_punctuation: When ``True``, common punctuation
                is kept.

        Returns:
            Cleaned text.
        """
        if preserve_punctuation:
            return self._SPECIAL_CHAR_RE.sub("", text)
        return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters.

        Args:
            text: Input text.

        Returns:
            Text with normalized whitespace.
        """
        return self._processor.normalize_whitespace(text)

    def remove_control_characters(self, text: str) -> str:
        """Remove Unicode control characters.

        Args:
            text: Input text.

        Returns:
            Text with control characters removed.
        """
        return self._processor.remove_unicode_control_chars(text)

    def normalize_unicode(self, text: str) -> str:
        """Apply Unicode NFC normalization.

        Args:
            text: Input text.

        Returns:
            NFC-normalized text.
        """
        return unicodedata.normalize("NFC", text)

    def standardize_encoding(
        self, text: str, target: str = "utf-8"
    ) -> str:
        """Re-encode text to the target encoding.

        Args:
            text: Input text.
            target: Target encoding (default ``utf-8``).

        Returns:
            Re-encoded text.
        """
        return self._processor.fix_encoding_issues(text)

    def full_normalize(self, text: str) -> str:
        """Run all normalization steps in order.

        Args:
            text: Raw input text.

        Returns:
            Fully normalized text.
        """
        text = self.standardize_encoding(text)
        text = self.normalize_unicode(text)
        text = self.remove_html_tags(text)
        text = self.remove_control_characters(text)
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)
        return text


class LanguageValidator:
    """Language detection and validation.

    Uses the ``langdetect`` library for language identification.
    Falls back gracefully when the library is unavailable or the
    text is too short for reliable detection.
    """

    def __init__(
        self,
        supported_languages: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self._supported = set(
            supported_languages or ["en"]
        )
        self._confidence_threshold = confidence_threshold
        self._last_confidence: float = 0.0

    def detect_language(self, text: str) -> str:
        """Detect the primary language of *text*.

        Args:
            text: Input text.

        Returns:
            ISO-639-1 language code, or ``"unknown"`` on failure.
        """
        try:
            from langdetect import detect_langs
            results = detect_langs(text)
            if results:
                top = results[0]
                self._last_confidence = top.prob
                return str(top.lang)
        except Exception:
            logger.debug(
                "Language detection failed; defaulting to 'unknown'"
            )
        self._last_confidence = 0.0
        return "unknown"

    def is_supported_language(self, language_code: str) -> bool:
        """Check if a language code is in the supported set.

        Args:
            language_code: ISO-639-1 code.

        Returns:
            ``True`` if supported.
        """
        return language_code in self._supported

    def get_language_confidence(self) -> float:
        """Return confidence of the most recent detection.

        Returns:
            Confidence in ``[0, 1]``.
        """
        return self._last_confidence

    def handle_multilingual_content(
        self, text: str
    ) -> Dict[str, float]:
        """Detect all languages present in *text*.

        Args:
            text: Input text.

        Returns:
            Dictionary mapping language codes to their probabilities.
        """
        try:
            from langdetect import detect_langs
            results = detect_langs(text)
            return {str(r.lang): r.prob for r in results}
        except Exception:
            logger.debug("Multilingual detection failed")
            return {}


class ContentValidator:
    """Content validation checks (length, tokens, coherence, spam)."""

    _SPAM_PATTERNS = [
        re.compile(r"(?i)(buy now|click here|free money|act now)"),
        re.compile(r"(.)\1{9,}"),  # repeated chars
        re.compile(
            r"(?i)(viagra|cialis|lottery|winner|congratulations"
            r"|nigerian prince)"
        ),
        re.compile(r"(?i)(special offer|limited time|order now)"),
        re.compile(r"[A-Z\s!]{20,}"),  # long UPPERCASE runs
    ]

    def __init__(self, config: Optional[Phase2Config] = None):
        self._config = config or Phase2Config()
        self._token_counter = TokenCounter()

    def validate_minimum_length(
        self, text: str, min_length: int = 10
    ) -> bool:
        """Check that text meets the minimum length.

        Args:
            text: Input text.
            min_length: Minimum character count.

        Returns:
            ``True`` if long enough.
        """
        return len(text.strip()) >= min_length

    def validate_maximum_length(
        self, text: str, max_length: int = 50000
    ) -> bool:
        """Check that text does not exceed the maximum length.

        Args:
            text: Input text.
            max_length: Maximum character count.

        Returns:
            ``True`` if within limit.
        """
        return len(text.strip()) <= max_length

    def validate_token_count(self, text: str) -> int:
        """Count the tokens in *text*.

        Args:
            text: Input text.

        Returns:
            Number of whitespace-separated tokens.
        """
        return self._token_counter.count_tokens(text)

    def check_coherence_score(self, text: str) -> float:
        """Compute a simple coherence score for *text*.

        The heuristic considers unique-word ratio, average word
        length, and the presence of sentence-ending punctuation.
        Values are in ``[0, 1]``.

        Args:
            text: Input text.

        Returns:
            Coherence score.
        """
        words = text.split()
        if not words:
            return 0.0

        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        avg_word_len = sum(len(w) for w in words) / len(words)
        length_score = min(avg_word_len / 8.0, 1.0)
        has_punctuation = float(
            bool(re.search(r"[.!?]", text))
        )

        return (
            0.4 * unique_ratio
            + 0.3 * length_score
            + 0.3 * has_punctuation
        )

    def detect_spam_patterns(self, text: str) -> bool:
        """Check whether *text* matches known spam patterns.

        Args:
            text: Input text.

        Returns:
            ``True`` if spam patterns are detected.
        """
        matches = sum(
            1 for pattern in self._SPAM_PATTERNS
            if pattern.search(text)
        )
        return matches >= 1

    def count_spam_matches(self, text: str) -> int:
        """Count spam pattern matches in *text*.

        Args:
            text: Input text.

        Returns:
            Number of distinct spam patterns matched.
        """
        return sum(
            1 for pattern in self._SPAM_PATTERNS
            if pattern.search(text)
        )

    def is_valid_content(self, text: str) -> dict:
        """Run all content validation checks.

        Args:
            text: Input text.

        Returns:
            Dictionary with boolean results keyed by check name.
        """
        return {
            "min_length": self.validate_minimum_length(
                text, self._config.input_min_length
            ),
            "max_length": self.validate_maximum_length(
                text, self._config.input_max_length
            ),
            "token_count": self.validate_token_count(text),
            "coherence": self.check_coherence_score(text),
            "is_spam": self.detect_spam_patterns(text),
        }


class DomainTagExtractor:
    """Extract domain tags from text using semantic clustering.

    When the ``auto_semantic_clusterer`` module is available in the
    Python path, it is used to map extracted keywords to canonical
    domains. Otherwise, a lightweight keyword-matching fallback is
    employed.
    """

    # Lightweight keyword-based fallback.
    _KEYWORD_DOMAINS: Dict[str, List[str]] = {
        "medical": [
            "medical", "medicine", "healthcare", "disease",
            "treatment", "patient", "clinical", "diagnosis",
            "therapy", "biology", "biomedical",
        ],
        "physics": [
            "physics", "quantum", "thermodynamics", "particle",
            "electromagnetic", "optics", "energy", "force",
        ],
        "chemistry": [
            "chemistry", "chemical", "molecule", "compound",
            "reaction", "organic", "inorganic", "biochemistry",
        ],
        "mathematics": [
            "mathematics", "algebra", "calculus", "geometry",
            "statistics", "equation", "theorem",
        ],
        "computer_science": [
            "computer", "programming", "algorithm", "software",
            "artificial intelligence", "machine learning",
        ],
        "music": [
            "music", "song", "melody", "harmony", "instrument",
            "composition", "rhythm",
        ],
    }

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
    ) -> None:
        self._config = config or Phase2Config()
        self._clusterer = None
        self._clusterer_available = False
        self._init_clusterer()

    def _init_clusterer(self) -> None:
        """Try to initialise the auto semantic clusterer."""
        if not self._config.use_auto_clustering:
            return
        try:
            # The module lives at the repository root, which may
            # not be on sys.path. Add it if necessary.
            import importlib
            spec = importlib.util.find_spec(
                "auto_semantic_clusterer"
            )
            if spec is None:
                import os
                repo_root = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)

            from auto_semantic_clusterer import (
                AutoSemanticClusterer,
            )
            self._clusterer = AutoSemanticClusterer()
            self._clusterer.similarity_threshold = (
                self._config.similarity_threshold
            )
            self._clusterer.initialize()
            self._clusterer_available = True
            logger.info("Auto semantic clusterer initialised")
        except Exception as exc:
            logger.warning(
                "Auto semantic clusterer unavailable (%s); "
                "using keyword fallback",
                exc,
            )
            self._clusterer_available = False

    def extract_tags(self, text: str) -> List[str]:
        """Extract candidate domain tags from *text*.

        Uses simple keyword matching against known domain terms.

        Args:
            text: Input text.

        Returns:
            List of matched keyword tags.
        """
        text_lower = text.lower()
        tags: List[str] = []
        for domain, keywords in self._KEYWORD_DOMAINS.items():
            for kw in keywords:
                if kw in text_lower and kw not in tags:
                    tags.append(kw)
        return tags[: self._config.max_tags]

    def filter_by_confidence(
        self, tags: List[str], threshold: float = 0.45
    ) -> List[str]:
        """Keep only tags that meet the confidence threshold.

        Args:
            tags: List of candidate tags.
            threshold: Minimum confidence score.

        Returns:
            Filtered list of tags.
        """
        scores = self.get_tag_confidence_scores(tags)
        return [
            t for t in tags if scores.get(t, 0.0) >= threshold
        ]

    def get_tag_confidence_scores(
        self, tags: List[str]
    ) -> Dict[str, float]:
        """Get confidence scores for each tag.

        When the auto semantic clusterer is available, confidence
        comes from cosine similarity. Otherwise, a fixed score of
        ``0.8`` is assigned to keyword matches.

        Args:
            tags: List of tags.

        Returns:
            Dictionary mapping each tag to its confidence score.
        """
        scores: Dict[str, float] = {}
        if self._clusterer_available and self._clusterer:
            results = self._clusterer.cluster_tags_batch(tags)
            for tag in tags:
                if tag in results:
                    _, conf = results[tag]
                    scores[tag] = conf
                else:
                    scores[tag] = 0.0
        else:
            for tag in tags:
                scores[tag] = 0.8
        return scores

    def map_tags_to_domains(
        self, tags: List[str]
    ) -> Dict[str, List[str]]:
        """Map tags to their canonical domain names.

        Args:
            tags: List of tags.

        Returns:
            Dictionary mapping domain names to lists of tags.
        """
        mapping: Dict[str, List[str]] = {}
        if self._clusterer_available and self._clusterer:
            results = self._clusterer.cluster_tags_batch(tags)
            for tag in tags:
                if tag in results:
                    domain, _ = results[tag]
                    if domain:
                        mapping.setdefault(domain, []).append(tag)
        else:
            for tag in tags:
                tag_lower = tag.lower()
                for domain, kws in self._KEYWORD_DOMAINS.items():
                    if tag_lower in kws:
                        mapping.setdefault(domain, []).append(tag)
                        break
        return mapping

    def integrate_with_auto_semantic_clusterer(self) -> bool:
        """Check if auto semantic clusterer integration is active.

        Returns:
            ``True`` if the clusterer is initialised and ready.
        """
        return self._clusterer_available


class EarlyRejectionFilter:
    """Determine whether input should be rejected early.

    Combines heuristic checks for spam, gibberish, off-topic
    content, and unsupported language.
    """

    _GIBBERISH_RE = re.compile(
        r"^[^a-zA-Z]*$|^(.)\1{19,}$"
    )

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
    ) -> None:
        self._config = config or Phase2Config()

    def should_reject(self, text: str) -> bool:
        """Check if *text* should be rejected.

        Args:
            text: Input text.

        Returns:
            ``True`` if the text should be rejected.
        """
        return (
            self.compute_rejection_score(text)
            >= self._config.rejection_score_threshold
        )

    def get_rejection_reason(self, text: str) -> str:
        """Get a human-readable reason for rejection.

        Args:
            text: Input text.

        Returns:
            A short description, or an empty string if no
            rejection reason applies.
        """
        reasons: List[str] = []

        stripped = text.strip()
        if not stripped:
            reasons.append("empty input")
        elif len(stripped) < self._config.input_min_length:
            reasons.append("text too short")

        content_validator = ContentValidator(self._config)
        if content_validator.detect_spam_patterns(text):
            reasons.append("spam detected")

        if self._is_gibberish(text):
            reasons.append("gibberish detected")

        return "; ".join(reasons) if reasons else ""

    def compute_rejection_score(self, text: str) -> float:
        """Compute a combined rejection score in ``[0, 1]``.

        Higher values indicate stronger evidence for rejection.

        Args:
            text: Input text.

        Returns:
            Rejection score.
        """
        score = 0.0
        stripped = text.strip()

        if not stripped:
            return 1.0

        if len(stripped) < self._config.input_min_length:
            score += 0.4

        content_validator = ContentValidator(self._config)
        spam_count = content_validator.count_spam_matches(text)
        if spam_count >= 2:
            score += 0.7
        elif spam_count == 1:
            score += 0.5

        if self._is_gibberish(text):
            score += 0.4

        coherence = content_validator.check_coherence_score(text)
        if coherence < self._config.coherence_min_score:
            score += 0.3

        return min(score, 1.0)

    def _is_gibberish(self, text: str) -> bool:
        """Quick check for gibberish text."""
        stripped = text.strip()
        if not stripped:
            return True
        if self._GIBBERISH_RE.match(stripped):
            return True
        words = stripped.split()
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len > 25:
                return True
        return False


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class InputNormalizationPipeline:
    """Orchestrates all Phase 2.1 sub-components.

    Usage:
        >>> pipeline = InputNormalizationPipeline()
        >>> result = pipeline.normalize("Some input text here.")
        >>> print(result.is_valid)
        True
    """

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
    ) -> None:
        self._config = config or Phase2Config()

        self._text_normalizer = TextNormalizer()
        self._language_validator = LanguageValidator(
            supported_languages=self._config.supported_languages,
            confidence_threshold=(
                self._config.language_confidence_threshold
            ),
        )
        self._content_validator = ContentValidator(self._config)
        self._tag_extractor = DomainTagExtractor(self._config)
        self._rejection_filter = EarlyRejectionFilter(self._config)

        self._input_validator = InputValidator()
        self._result_validator = ResultValidator()

        self._last_metadata: dict = {}

        # Configure logging
        logging.basicConfig(
            level=getattr(
                logging, self._config.log_level.upper(), logging.INFO
            ),
            format=(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            ),
        )

    # ---- public API ----

    def normalize(self, raw_input: str) -> NormalizationResult:
        """Run the full normalization pipeline on *raw_input*.

        Args:
            raw_input: The raw user input string.

        Returns:
            A ``NormalizationResult`` with all fields populated.

        Raises:
            InvalidInputTypeError: If *raw_input* is not a string.
        """
        start = time.perf_counter()
        result = NormalizationResult(original_text=raw_input)
        steps_executed: List[str] = []

        try:
            # Step 0: type check
            if not self._input_validator.validate_string_type(
                raw_input
            ):
                raise InvalidInputTypeError(
                    f"Expected str, got {type(raw_input).__name__}"
                )
            steps_executed.append("type_check")

            # Step 1: text normalization
            cleaned = self._text_normalizer.full_normalize(raw_input)
            result.cleaned_text = cleaned
            steps_executed.append("text_normalization")
            logger.debug("Cleaned text length: %d", len(cleaned))

            # Step 2: language detection
            lang = self._language_validator.detect_language(cleaned)
            result.language = lang
            result.language_confidence = (
                self._language_validator.get_language_confidence()
            )
            steps_executed.append("language_detection")

            if not self._language_validator.is_supported_language(
                lang
            ):
                result.warnings.append(
                    f"Unsupported language detected: {lang}"
                )

            # Step 3: content validation
            validation = self._content_validator.is_valid_content(
                cleaned
            )
            result.validation_report = validation
            steps_executed.append("content_validation")

            is_valid = (
                validation["min_length"]
                and validation["max_length"]
                and not validation["is_spam"]
            )
            result.is_valid = is_valid

            # Step 4: domain tag extraction
            tags = self._tag_extractor.extract_tags(cleaned)
            result.extracted_tags = tags
            result.tag_confidence_scores = (
                self._tag_extractor.get_tag_confidence_scores(tags)
            )
            result.domain_mapping = (
                self._tag_extractor.map_tags_to_domains(tags)
            )
            steps_executed.append("tag_extraction")

            # Step 5: early rejection
            result.should_reject = (
                self._rejection_filter.should_reject(cleaned)
            )
            result.rejection_reason = (
                self._rejection_filter.get_rejection_reason(cleaned)
            )
            result.rejection_score = (
                self._rejection_filter.compute_rejection_score(
                    cleaned
                )
            )
            steps_executed.append("early_rejection")

            if result.should_reject:
                result.is_valid = False

        except InvalidInputTypeError:
            result.is_valid = False
            result.should_reject = True
            result.rejection_reason = "invalid input type"
            result.rejection_score = 1.0
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in normalization pipeline: %s",
                exc,
            )
            result.is_valid = False
            result.warnings.append(f"Pipeline error: {exc}")
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.processing_time_ms = elapsed_ms
            self._last_metadata = {
                "processing_time_ms": elapsed_ms,
                "steps_executed": steps_executed,
            }

        return result

    def validate_end_to_end(self, raw_input: str) -> bool:
        """Run the pipeline and return a simple pass/fail.

        Args:
            raw_input: The raw user input string.

        Returns:
            ``True`` if the input is valid and should not be
            rejected.
        """
        try:
            result = self.normalize(raw_input)
            return result.is_valid and not result.should_reject
        except InputNormalizationError:
            return False

    def get_processing_metadata(self) -> dict:
        """Return metadata from the last ``normalize`` call.

        Returns:
            Dictionary with ``processing_time_ms`` and
            ``steps_executed``.
        """
        return dict(self._last_metadata)
