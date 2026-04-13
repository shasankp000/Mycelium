"""
Shared pytest fixtures for Phase 2 tests.
"""

import pytest


@pytest.fixture
def sample_clean_text():
    """A well-formed English sentence."""
    return "The mitochondria is the powerhouse of the cell."


@pytest.fixture
def sample_html_text():
    """Text containing HTML tags and entities."""
    return (
        "<p>This is a <b>bold</b> statement about &amp; "
        "medical <i>treatment</i>.</p>"
    )


@pytest.fixture
def sample_special_chars_text():
    """Text with special and unusual characters."""
    return "Hello™ world© — this «text» has ‡special‡ chars!"


@pytest.fixture
def sample_unicode_text():
    """Text with Unicode characters and accents."""
    return "Ça fait résumé naïve über café"


@pytest.fixture
def sample_multilingual_text():
    """Text mixing English and another language."""
    return (
        "This is an English sentence. "
        "Dies ist ein deutscher Satz."
    )


@pytest.fixture
def sample_spam_text():
    """Text that matches spam patterns."""
    return (
        "CONGRATULATIONS! You are the WINNER! "
        "Click here to claim your FREE MONEY now! "
        "Buy now and get a special offer! Act now!"
    )


@pytest.fixture
def sample_gibberish_text():
    """Text that is clearly gibberish."""
    return "asdfghjklqwertyuiopzxcvbnm"


@pytest.fixture
def sample_medical_text():
    """Domain-specific text about medicine."""
    return (
        "Metastatic carcinoma requires systemic chemotherapy "
        "rather than localized radiation treatment. The patient "
        "was diagnosed with stage IV cancer and the clinical "
        "team recommended immunotherapy as the primary therapy."
    )


@pytest.fixture
def sample_short_text():
    """Text that is too short to be valid."""
    return "Hi"


@pytest.fixture
def sample_long_text():
    """Text that is very long but valid."""
    return "This is a valid sentence. " * 200


@pytest.fixture
def expected_normalization_result():
    """Expected fields of a NormalizationResult."""
    return {
        "cleaned_text": str,
        "original_text": str,
        "language": str,
        "language_confidence": float,
        "is_valid": bool,
        "extracted_tags": list,
        "tag_confidence_scores": dict,
        "domain_mapping": dict,
        "should_reject": bool,
        "rejection_reason": (str, type(None)),
        "rejection_score": float,
        "processing_time_ms": float,
        "validation_report": dict,
        "warnings": list,
    }
