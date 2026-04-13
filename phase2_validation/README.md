# Phase 2 Validation Pipeline

Input normalization and pre-filtering layer for the Mycelium Expert System's six-phase reasoning pipeline.

## Overview

This package implements **Phase 2.1 — Input Normalization & Pre-filtering**, the first stage of the reasoning pipeline. It cleans, validates, and enriches raw user input before downstream phases perform semantic understanding, expert selection, inference, calibration, and decision synthesis.

### Components

| Component | Description |
|---|---|
| `TextNormalizer` | HTML removal, Unicode normalization, encoding fixes |
| `LanguageValidator` | Language detection with confidence scores |
| `ContentValidator` | Length, token count, coherence, and spam checks |
| `DomainTagExtractor` | Keyword/semantic tag extraction mapped to domains |
| `EarlyRejectionFilter` | Spam, gibberish, and off-topic rejection |
| `InputNormalizationPipeline` | Orchestrator that runs all steps end-to-end |

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `langdetect` — language detection
- `ftfy` — Unicode and encoding fixes
- `bleach` — HTML sanitization
- `nltk` — natural language processing utilities
- `PyYAML` — YAML configuration loading (optional)

## Configuration

All settings live in `phase2_validation/config/phase2_config.py` and can be overridden via environment variables prefixed with `P2_`:

| Variable | Default | Description |
|---|---|---|
| `P2_INPUT_MIN_LENGTH` | `10` | Minimum character length |
| `P2_INPUT_MAX_LENGTH` | `50000` | Maximum character length |
| `P2_INPUT_MIN_TOKENS` | `3` | Minimum token count |
| `P2_INPUT_MAX_TOKENS` | `10000` | Maximum token count |
| `P2_LANG_CONFIDENCE_THRESHOLD` | `0.5` | Language detection confidence threshold |
| `P2_SIMILARITY_THRESHOLD` | `0.45` | Domain tag cosine similarity threshold |
| `P2_TAG_CONFIDENCE_THRESHOLD` | `0.45` | Tag confidence filter threshold |
| `P2_SPAM_SCORE_THRESHOLD` | `0.7` | Spam score threshold |
| `P2_REJECTION_SCORE_THRESHOLD` | `0.6` | Combined rejection score threshold |
| `P2_COHERENCE_MIN_SCORE` | `0.3` | Minimum coherence score |
| `P2_LOG_LEVEL` | `INFO` | Logging level |
| `P2_USE_AUTO_CLUSTERING` | `True` | Use auto semantic clusterer |

You can also load configuration from a YAML file:

```python
from phase2_validation.config.phase2_config import Phase2Config

config = Phase2Config.from_yaml("my_config.yaml")
```

## Usage

### Basic Pipeline Usage

```python
from phase2_validation.phases.phase_2_1_input_normalization import (
    InputNormalizationPipeline,
)

pipeline = InputNormalizationPipeline()
result = pipeline.normalize("Patient diagnosed with stage IV cancer.")

print(result.is_valid)              # True
print(result.cleaned_text)          # "Patient diagnosed with stage IV cancer."
print(result.language)              # "en"
print(result.extracted_tags)        # ["patient", "diagnosis", ...]
print(result.domain_mapping)        # {"medical": ["patient", ...]}
print(result.processing_time_ms)    # 12.5
```

### Quick Validation

```python
is_ok = pipeline.validate_end_to_end("Some user input here.")
# Returns True / False
```

### Processing Metadata

```python
pipeline.normalize("Hello world, this is a test input.")
meta = pipeline.get_processing_metadata()
print(meta)
# {"processing_time_ms": 8.2, "steps_executed": ["type_check", ...]}
```

## Testing

```bash
# Run all Phase 2 tests
pytest tests/phase2/ -v

# With coverage
pytest tests/phase2/ --cov=phase2_validation --cov-report=term-missing
```

## Directory Structure

```
phase2_validation/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── phase2_config.py
├── phases/
│   ├── __init__.py
│   ├── phase_2_1_input_normalization.py
│   ├── phase_2_2_semantic_understanding.py (stub)
│   ├── phase_2_3_expert_selection.py (stub)
│   ├── phase_2_4_inference.py (stub)
│   ├── phase_2_5_calibration.py (stub)
│   └── phase_2_6_synthesis.py (stub)
├── utils/
│   ├── __init__.py
│   ├── text_processors.py
│   └── validators.py
├── pipeline.py (stub)
└── README.md
```

## Performance

Typical processing times on a modern laptop:

| Step | Time |
|---|---|
| Text normalization | < 1 ms |
| Language detection | 5–15 ms |
| Content validation | < 1 ms |
| Tag extraction (keyword) | < 1 ms |
| Tag extraction (semantic) | 20–50 ms |
| Early rejection check | < 1 ms |
| **Full pipeline** | **10–70 ms** |
