"""
Tests for the Phase 2.1 Input Normalization Pipeline.
"""

import pytest

from phase2_validation.config.phase2_config import Phase2Config
from phase2_validation.phases.phase_2_1_input_normalization import (
    ContentValidator,
    DomainTagExtractor,
    EarlyRejectionFilter,
    InputNormalizationPipeline,
    InvalidInputTypeError,
    LanguageValidator,
    NormalizationResult,
    TextNormalizer,
)


# ------------------------------------------------------------------
# TextNormalizer
# ------------------------------------------------------------------


class TestTextNormalizer:
    def setup_method(self):
        self.normalizer = TextNormalizer()

    def test_remove_html_tags(self):
        result = self.normalizer.remove_html_tags(
            "<p>Hello <b>world</b></p>"
        )
        assert "<" not in result
        assert "Hello" in result

    def test_remove_special_characters(self):
        result = self.normalizer.remove_special_characters(
            "Hello™ world©"
        )
        assert "™" not in result
        assert "Hello" in result

    def test_normalize_whitespace(self):
        result = self.normalizer.normalize_whitespace(
            "  hello   world  "
        )
        assert result == "hello world"

    def test_remove_control_characters(self):
        result = self.normalizer.remove_control_characters(
            "hello\x00world\x01test"
        )
        assert "\x00" not in result
        assert "\x01" not in result
        assert "helloworld" in result

    def test_normalize_unicode(self):
        import unicodedata

        text = "caf\u0065\u0301"
        result = self.normalizer.normalize_unicode(text)
        assert unicodedata.is_normalized("NFC", result)

    def test_standardize_encoding(self):
        result = self.normalizer.standardize_encoding(
            "hello world"
        )
        assert isinstance(result, str)

    def test_full_normalize(self):
        raw = "<p>  Hello\x00  <b>world</b>  </p>"
        result = self.normalizer.full_normalize(raw)
        assert "<" not in result
        assert "\x00" not in result
        assert result == result.strip()


# ------------------------------------------------------------------
# LanguageValidator
# ------------------------------------------------------------------


class TestLanguageValidator:
    def setup_method(self):
        self.validator = LanguageValidator()

    def test_detect_language(self):
        lang = self.validator.detect_language(
            "This is an English sentence about biology."
        )
        assert isinstance(lang, str)

    def test_is_supported_language(self):
        assert self.validator.is_supported_language("en")
        assert not self.validator.is_supported_language("xx")

    def test_get_language_confidence(self):
        self.validator.detect_language(
            "This is clearly an English sentence."
        )
        conf = self.validator.get_language_confidence()
        assert 0.0 <= conf <= 1.0

    def test_handle_multilingual_content(self):
        result = self.validator.handle_multilingual_content(
            "This is English. Dies ist Deutsch."
        )
        assert isinstance(result, dict)


# ------------------------------------------------------------------
# ContentValidator
# ------------------------------------------------------------------


class TestContentValidator:
    def setup_method(self):
        self.validator = ContentValidator()

    def test_validate_minimum_length(self):
        assert self.validator.validate_minimum_length(
            "A" * 20, min_length=10
        )
        assert not self.validator.validate_minimum_length(
            "Hi", min_length=10
        )

    def test_validate_maximum_length(self):
        assert self.validator.validate_maximum_length(
            "Short text", max_length=1000
        )
        assert not self.validator.validate_maximum_length(
            "A" * 100, max_length=10
        )

    def test_validate_token_count(self):
        count = self.validator.validate_token_count(
            "one two three four"
        )
        assert count == 4

    def test_check_coherence_score(self):
        score = self.validator.check_coherence_score(
            "The patient was diagnosed with cancer."
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_detect_spam_patterns(self):
        assert self.validator.detect_spam_patterns(
            "Buy now and get free money!"
        )
        assert not self.validator.detect_spam_patterns(
            "Quantum mechanics is a branch of physics."
        )

    def test_is_valid_content(self):
        report = self.validator.is_valid_content(
            "A valid sentence about physics and energy."
        )
        assert isinstance(report, dict)
        assert "min_length" in report
        assert "is_spam" in report


# ------------------------------------------------------------------
# DomainTagExtractor
# ------------------------------------------------------------------


class TestDomainTagExtractor:
    def setup_method(self):
        config = Phase2Config(use_auto_clustering=False)
        self.extractor = DomainTagExtractor(config=config)

    def test_extract_tags_medical(self):
        tags = self.extractor.extract_tags(
            "The patient needs medical treatment and therapy."
        )
        assert isinstance(tags, list)
        assert len(tags) > 0

    def test_extract_tags_empty(self):
        tags = self.extractor.extract_tags(
            "Nothing relevant here at all."
        )
        assert isinstance(tags, list)

    def test_get_tag_confidence_scores(self):
        tags = ["medical", "treatment"]
        scores = self.extractor.get_tag_confidence_scores(tags)
        assert isinstance(scores, dict)
        for tag in tags:
            assert tag in scores

    def test_filter_by_confidence(self):
        tags = ["medical", "unknown_xyz"]
        filtered = self.extractor.filter_by_confidence(
            tags, threshold=0.5
        )
        assert isinstance(filtered, list)

    def test_map_tags_to_domains(self):
        tags = ["medical", "physics"]
        mapping = self.extractor.map_tags_to_domains(tags)
        assert isinstance(mapping, dict)

    def test_integrate_with_auto_semantic_clusterer(self):
        result = (
            self.extractor.integrate_with_auto_semantic_clusterer()
        )
        assert isinstance(result, bool)


# ------------------------------------------------------------------
# EarlyRejectionFilter
# ------------------------------------------------------------------


class TestEarlyRejectionFilter:
    def setup_method(self):
        self.filter = EarlyRejectionFilter()

    def test_should_reject_empty(self):
        assert self.filter.should_reject("")

    def test_should_reject_spam(self, sample_spam_text):
        assert self.filter.should_reject(sample_spam_text)

    def test_should_not_reject_valid(self):
        assert not self.filter.should_reject(
            "Quantum mechanics describes the behaviour of "
            "particles at the atomic and subatomic levels."
        )

    def test_get_rejection_reason(self):
        reason = self.filter.get_rejection_reason("")
        assert "empty" in reason.lower()

    def test_compute_rejection_score(self):
        score = self.filter.compute_rejection_score(
            "A normal sentence about science."
        )
        assert 0.0 <= score <= 1.0


# ------------------------------------------------------------------
# InputNormalizationPipeline
# ------------------------------------------------------------------


class TestInputNormalizationPipeline:
    def setup_method(self):
        config = Phase2Config(use_auto_clustering=False)
        self.pipeline = InputNormalizationPipeline(config=config)

    def test_pipeline_initialization(self):
        assert self.pipeline is not None

    def test_normalize_basic_text(self, sample_clean_text):
        result = self.pipeline.normalize(sample_clean_text)
        assert isinstance(result, NormalizationResult)
        assert result.is_valid
        assert result.cleaned_text

    def test_normalize_with_html(self, sample_html_text):
        result = self.pipeline.normalize(sample_html_text)
        assert "<" not in result.cleaned_text
        assert isinstance(result, NormalizationResult)

    def test_normalize_with_special_chars(
        self, sample_special_chars_text
    ):
        result = self.pipeline.normalize(
            sample_special_chars_text
        )
        assert isinstance(result, NormalizationResult)
        assert result.cleaned_text

    def test_normalize_with_whitespace_issues(self):
        text = "   Too   much   whitespace   everywhere   "
        result = self.pipeline.normalize(text)
        assert "  " not in result.cleaned_text

    def test_normalize_with_unicode(self, sample_unicode_text):
        result = self.pipeline.normalize(sample_unicode_text)
        assert isinstance(result, NormalizationResult)

    def test_normalize_with_control_characters(self):
        text = "Hello\x00world\x01this\x02is\x03a test sentence."
        result = self.pipeline.normalize(text)
        assert "\x00" not in result.cleaned_text

    def test_normalize_multilingual_text(
        self, sample_multilingual_text
    ):
        result = self.pipeline.normalize(
            sample_multilingual_text
        )
        assert isinstance(result, NormalizationResult)

    def test_normalize_empty_input(self):
        result = self.pipeline.normalize("")
        assert not result.is_valid

    def test_normalize_very_long_text(self, sample_long_text):
        result = self.pipeline.normalize(sample_long_text)
        assert isinstance(result, NormalizationResult)

    def test_normalize_spam_detection(self, sample_spam_text):
        result = self.pipeline.normalize(sample_spam_text)
        assert result.should_reject or not result.is_valid

    def test_normalize_with_all_issues_combined(self):
        text = (
            "  <p>Buy now!!!  \x00  &amp; get "
            "FREE™ \u201cstuff\u201d  </p>  "
        )
        result = self.pipeline.normalize(text)
        assert isinstance(result, NormalizationResult)
        assert "<" not in result.cleaned_text

    def test_domain_tag_extraction(self, sample_medical_text):
        result = self.pipeline.normalize(sample_medical_text)
        assert isinstance(result.extracted_tags, list)

    def test_tag_confidence_filtering(self, sample_medical_text):
        result = self.pipeline.normalize(sample_medical_text)
        assert isinstance(result.tag_confidence_scores, dict)

    def test_processing_metadata_collection(
        self, sample_clean_text
    ):
        self.pipeline.normalize(sample_clean_text)
        metadata = self.pipeline.get_processing_metadata()
        assert "processing_time_ms" in metadata
        assert "steps_executed" in metadata
        assert metadata["processing_time_ms"] > 0

    def test_idempotency_of_normalization(
        self, sample_clean_text
    ):
        r1 = self.pipeline.normalize(sample_clean_text)
        r2 = self.pipeline.normalize(r1.cleaned_text)
        assert r1.cleaned_text == r2.cleaned_text

    def test_error_handling_invalid_input(self):
        with pytest.raises(InvalidInputTypeError):
            self.pipeline.normalize(12345)  # type: ignore[arg-type]

    def test_end_to_end_validation(self, sample_clean_text):
        valid = self.pipeline.validate_end_to_end(
            sample_clean_text
        )
        assert isinstance(valid, bool)

    def test_end_to_end_validation_invalid(self):
        assert not self.pipeline.validate_end_to_end("")

    def test_result_fields(
        self, sample_clean_text, expected_normalization_result
    ):
        result = self.pipeline.normalize(sample_clean_text)
        for field_name, expected_type in (
            expected_normalization_result.items()
        ):
            assert hasattr(result, field_name), (
                f"Missing field: {field_name}"
            )
            value = getattr(result, field_name)
            if isinstance(expected_type, tuple):
                assert isinstance(value, expected_type), (
                    f"{field_name}: expected {expected_type}, "
                    f"got {type(value)}"
                )
            else:
                assert isinstance(value, expected_type), (
                    f"{field_name}: expected {expected_type}, "
                    f"got {type(value)}"
                )
