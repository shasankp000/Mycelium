"""
dataset_logger.py
==================
Thread-safe live query logger for Layer 0.

Every time the LLM arbiter fires for any Layer 0 component, it writes one
JSON-lines record to:

    <project_root>/training_data/dataset_log.jsonl

This file is consumed by:
    - train_layer0_models.py   — offline trainer (reads it at training time)
    - finetune_layer0_models.py (future) — incremental fine-tuning script

Schema per record
-----------------
{
  "ts"              : "2026-05-27T13:45:00.123456",   # ISO-8601 UTC
  "component"       : "manipulation" | "objectivity" | "assumption",
  "text"            : "<raw query text>",
  "llm_label"       : "<label string>",
  "confidence"      : 0.87,                           # float 0.0-1.0
  "rule_score"      : 0.45,                           # float, optional
  "matched_signals" : ["jailbreak_structural", ...],  # list[str], optional
  "sentence_type"   : "INTERROGATIVE",                # optional
  "assumption_types": ["FACTIVE_PRESUPPOSITION", ...] # only for 'assumption' component
}

The file is append-only and never truncated by this module.  Each call
requires the global write lock so concurrent workers never interleave bytes.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "training_data" / "dataset_log.jsonl"

# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_entry(
    component:       str,
    text:            str,
    llm_label:       str,
    confidence:      float,
    rule_score:      float = 0.0,
    matched_signals: Optional[List[str]] = None,
    sentence_type:   Optional[str] = None,
    assumption_types: Optional[List[str]] = None,
    **extra: Any,
) -> None:
    """
    Append one labelled record to training_data/dataset_log.jsonl.

    Parameters
    ----------
    component        : Which Layer 0 component fired the LLM
                       ("manipulation" | "objectivity" | "assumption").
    text             : The raw query / input string that was analysed.
    llm_label        : The label returned by the LLM arbiter.
    confidence       : Confidence score from the LLM response (0.0-1.0).
    rule_score       : Rule-based score before the LLM was invoked.
    matched_signals  : List of rule signal names that triggered.
    sentence_type    : Detected sentence type (INTERROGATIVE / DECLARATIVE …).
    assumption_types : List of assumption type strings (assumption component only).
    **extra          : Any additional key-value pairs to include in the record.
    """
    record: Dict[str, Any] = {
        "ts":         datetime.now(tz=timezone.utc).isoformat(),
        "component":  component,
        "text":       text,
        "llm_label":  llm_label,
        "confidence": round(float(confidence), 4),
    }
    if rule_score:
        record["rule_score"] = round(float(rule_score), 4)
    if matched_signals:
        record["matched_signals"] = list(matched_signals)
    if sentence_type:
        record["sentence_type"] = sentence_type
    if assumption_types:
        record["assumption_types"] = list(assumption_types)
    record.update(extra)

    line = json.dumps(record, ensure_ascii=False)

    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:
        # Logging must never crash the main pipeline.
        logger.warning("dataset_logger: failed to write record: %s", exc)


def log_path() -> Path:
    """Return the path to the JSONL log file (does not check existence)."""
    return _LOG_PATH


def entry_count() -> int:
    """Return the number of logged entries (0 if file does not exist)."""
    if not _LOG_PATH.exists():
        return 0
    count = 0
    try:
        with open(_LOG_PATH, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
    except Exception:
        pass
    return count
