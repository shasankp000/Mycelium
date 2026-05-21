"""
patch_batch_logger.py

Thread-safe logger for CREATE_NEW_PATCH events.

Every time the pipeline decides CREATE_NEW_PATCH (i.e. no domain expert is
available for the query), this module:

  1. Buffers the JSONL record in memory (keyed by trace_id).
  2. Flushes to disk when fill_response() is called (response is ready),
     or when the buffer reaches 100 pending entries, or on process exit.

This removes the per-query open/write/close that previously occurred on
every log_query() call, eliminating both the I/O overhead and the race
condition under concurrent API load.

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

import atexit
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_BATCH_DIR = Path(os.getenv("PATCH_BATCH_DIR", "patch_batches"))
_DATE_FMT = "%Y-%m-%d"
_FLUSH_THRESHOLD = 100  # flush to disk when this many completed records accumulate


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

    Write strategy (M4 item 6.2)
    -----------------------------
    - ``log_query()`` only writes to the in-memory ``_pending`` dict — no disk I/O.
    - ``fill_response()`` moves the completed record into ``_ready`` and calls
      ``_maybe_flush()``.
    - ``_maybe_flush()`` writes all ``_ready`` records to disk when the buffer
      reaches ``_FLUSH_THRESHOLD`` entries or when called explicitly.
    - ``_flush_all()`` is registered with ``atexit`` so no records are lost on
      normal process exit.
    - Both ``_pending`` and ``_ready`` are protected by a single ``threading.Lock``.
    """

    def __init__(self, batch_dir: Path = _BATCH_DIR) -> None:
        self._batch_dir = batch_dir
        self._batch_dir.mkdir(parents=True, exist_ok=True)
        # Records waiting for fill_response() (query logged, response pending)
        self._pending: Dict[str, Dict[str, Any]] = {}
        # Records whose response has arrived and are ready to flush to disk
        self._ready: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        atexit.register(self._flush_all)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_query(
        self,
        *,
        trace_id: str,
        query: str,
        tags: Optional[List[str]] = None,
        layer0_route: Optional[str] = None,
        routing_classification: Optional[str] = None,
        domains: Optional[List[str]] = None,
        expert_decision: Optional[str] = None,
        confidence: Optional[float] = None,
        phase_latencies_ms: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Buffer a CREATE_NEW_PATCH event.

        No disk I/O is performed here.  The record is held in memory until
        :meth:`fill_response` marks it complete.
        """
        record: Dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "routing": {
                "layer0_route": layer0_route or "REASONING_PIPELINE",
                "classification": routing_classification or "",
                "domains": domains or [],
                "expert_decision": expert_decision or "CREATE_NEW_PATCH",
                "confidence": float(confidence or 0.0),
            },
            "tags": tags or [],
            "sandbox_evidence": [],
            "response": "",
            "phase_latencies_ms": phase_latencies_ms or {},
            "metadata": metadata or {},
        }
        with self._lock:
            self._pending[trace_id] = record

    def fill_response(self, trace_id: str, response: str) -> None:
        """Patch the ``response`` field and move the record to the flush queue.

        Triggers a disk flush if the ready buffer has reached the threshold.
        """
        with self._lock:
            record = self._pending.pop(trace_id, None)
            if record is None:
                # Already flushed or unknown trace — nothing to do.
                return
            record["response"] = response
            self._ready.append(record)
            should_flush = len(self._ready) >= _FLUSH_THRESHOLD

        if should_flush:
            self._flush_ready()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _today_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime(_DATE_FMT)
        return self._batch_dir / f"{date_str}.jsonl"

    def _flush_ready(self) -> None:
        """Write all records currently in ``_ready`` to today's JSONL file."""
        with self._lock:
            if not self._ready:
                return
            to_write = self._ready[:]
            self._ready.clear()

        if not to_write:
            return

        path = self._today_path()
        try:
            with path.open("a", encoding="utf-8") as fh:
                for record in to_write:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            # Re-queue on failure so records are not silently dropped
            with self._lock:
                self._ready[:0] = to_write
            raise exc

    def _flush_all(self) -> None:
        """Flush both the ready queue and any still-pending (no-response) records.

        Called automatically on process exit via ``atexit``.  Pending records
        are written with an empty ``response`` field so they are not lost.
        """
        with self._lock:
            orphans = list(self._pending.values())
            self._pending.clear()
            combined = self._ready + orphans
            self._ready.clear()

        if not combined:
            return

        path = self._today_path()
        try:
            with path.open("a", encoding="utf-8") as fh:
                for record in combined:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"⚠️  PatchBatchLogger: failed to flush on exit: {exc}")


# Process-wide singleton
patch_logger = PatchBatchLogger()
