"""
patch_batch_logger.py

Thread-safe logger for CREATE_NEW_PATCH events.

Every time the pipeline decides CREATE_NEW_PATCH (i.e. no domain expert is
available for the query), this module:

  1. Immediately appends a JSONL record containing the query + metadata to
     ``patch_batches/<YYYY-MM-DD>.jsonl``.
  2. Later, once the LLM has produced a response, ``fill_response()`` finds
     the record by ``trace_id`` and patches in the ``response`` field.

The daily JSONL files are consumed by the offline patch-model training
pipeline to build new domain experts from accumulated data.

Schema of each record
---------------------
::

    {
        "trace_id":               str,   # UUID v4 from the workflow run
        "timestamp":              str,   # ISO-8601 UTC
        "query":                  str,   # raw user input
        "tags":                   list,  # normalised tags from Layer 1
        "routing_classification": str,   # e.g. "ATTRIBUTE_ONLY"
        "response":               str,   # filled after LLM generation
        "phase_latencies_ms":     dict,  # per-phase timing
        "metadata":               dict   # arbitrary extra fields
    }
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_BATCH_DIR = Path(os.getenv("PATCH_BATCH_DIR", "patch_batches"))
_DATE_FMT = "%Y-%m-%d"


class PatchBatchLogger:
    """Append-only JSONL logger for CREATE_NEW_PATCH events.

    A single process-wide instance is created at module level and exported
    as ``patch_logger``.  Import and use it directly::

        from patch_batch_logger import patch_logger

        patch_logger.log_query(trace_id=..., query=..., tags=...,
                               routing_classification=...,
                               phase_latencies_ms=...)
        # … pipeline runs …
        patch_logger.fill_response(trace_id=..., response=final_answer)
    """

    def __init__(self, batch_dir: Path = _BATCH_DIR) -> None:
        self._batch_dir = batch_dir
        self._batch_dir.mkdir(parents=True, exist_ok=True)
        # In-flight records keyed by trace_id (awaiting response)
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_query(
        self,
        *,
        trace_id: str,
        query: str,
        tags: Optional[List[str]] = None,
        routing_classification: Optional[str] = None,
        phase_latencies_ms: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a CREATE_NEW_PATCH event immediately.

        The ``response`` field is left empty and filled later by
        :meth:`fill_response`.
        """
        record: Dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "tags": tags or [],
            "routing_classification": routing_classification or "",
            "response": "",
            "phase_latencies_ms": phase_latencies_ms or {},
            "metadata": metadata or {},
        }
        with self._lock:
            self._pending[trace_id] = record
            self._append(record)

    def fill_response(self, trace_id: str, response: str) -> None:
        """Patch the ``response`` field of a previously logged record.

        Rewrites the daily JSONL file with the updated record in-place.  For
        the typical single-writer, low-volume case this is acceptable; a
        database backend can replace this if volume grows.
        """
        with self._lock:
            record = self._pending.get(trace_id)
            if record is None:
                # Already flushed or unknown trace — nothing to do.
                return
            record["response"] = response
            self._pending.pop(trace_id, None)
            self._rewrite_record(trace_id, record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _today_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime(_DATE_FMT)
        return self._batch_dir / f"{date_str}.jsonl"

    def _append(self, record: Dict[str, Any]) -> None:
        """Append a single JSON line to today's batch file."""
        path = self._today_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _rewrite_record(
        self, trace_id: str, updated: Dict[str, Any]
    ) -> None:
        """Scan the daily file and replace the line matching trace_id."""
        path = self._today_path()
        if not path.exists():
            return

        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                new_lines.append(line)
                continue
            if obj.get("trace_id") == trace_id:
                new_lines.append(
                    json.dumps(updated, ensure_ascii=False) + "\n"
                )
            else:
                new_lines.append(line)

        path.write_text("".join(new_lines), encoding="utf-8")


# Process-wide singleton
patch_logger = PatchBatchLogger()
