"""
Tests for the validation utilities.
"""

import pytest

from phase2_validation.utils.validators import (
    InputValidator,
    ResultValidator,
)
from phase2_validation.phases.phase_2_1_input_normalization import (
    NormalizationResult,
)


class TestInputValidator:
    def setup_method(self):
        self.v = InputValidator()

    def test_validate_string_type(self):
        assert self.v.validate_string_type("hello")
        assert not self.v.validate_string_type(123)
        assert not self.v.validate_string_type(None)
        assert not self.v.validate_string_type([])

    def test_validate_encoding(self):
        assert self.v.validate_encoding("hello")
        assert self.v.validate_encoding("café")
        assert not self.v.validate_encoding(123)  # type: ignore[arg-type]

    def test_validate_not_empty(self):
        assert self.v.validate_not_empty("hello")
        assert not self.v.validate_not_empty("")
        assert not self.v.validate_not_empty("   ")
        assert not self.v.validate_not_empty(None)  # type: ignore[arg-type]

    def test_validate_length_range(self):
        assert self.v.validate_length_range(
            "A" * 20, min_len=10, max_len=100
        )
        assert not self.v.validate_length_range(
            "Hi", min_len=10, max_len=100
        )
        assert not self.v.validate_length_range(
            "A" * 200, min_len=10, max_len=100
        )

    def test_validate_token_range(self):
        assert self.v.validate_token_range(
            "one two three four five",
            min_tokens=3,
            max_tokens=10,
        )
        assert not self.v.validate_token_range(
            "one",
            min_tokens=3,
            max_tokens=10,
        )

    def test_edge_cases_and_boundary_conditions(self):
        # Exactly at min boundary
        assert self.v.validate_length_range(
            "A" * 10, min_len=10, max_len=100
        )
        # Exactly at max boundary
        assert self.v.validate_length_range(
            "A" * 100, min_len=10, max_len=100
        )
        # Non-string
        assert not self.v.validate_length_range(
            123, min_len=1, max_len=100  # type: ignore[arg-type]
        )


class TestResultValidator:
    def setup_method(self):
        self.v = ResultValidator()

    def test_validate_normalization_result(self):
        result = NormalizationResult(
            cleaned_text="hello",
            original_text="hello",
        )
        assert self.v.validate_normalization_result(result)

    def test_validate_normalization_result_invalid(self):
        assert not self.v.validate_normalization_result(
            {"not": "a dataclass"}
        )

    def test_validate_tag_extraction(self):
        assert self.v.validate_tag_extraction(
            ["tag1", "tag2"]
        )
        assert self.v.validate_tag_extraction([])
        assert not self.v.validate_tag_extraction("not a list")
        assert not self.v.validate_tag_extraction([1, 2, 3])

    def test_validate_domain_mapping(self):
        assert self.v.validate_domain_mapping(
            {"medical": ["disease", "treatment"]}
        )
        assert self.v.validate_domain_mapping({})
        assert not self.v.validate_domain_mapping("not a dict")
        assert not self.v.validate_domain_mapping(
            {1: ["bad key"]}
        )
        assert not self.v.validate_domain_mapping(
            {"key": "not a list"}
        )
