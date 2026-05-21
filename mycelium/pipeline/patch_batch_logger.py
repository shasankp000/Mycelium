"""
patch_batch_logger.py

Thread-safe logger for CREATE_NEW_PATCH events.

Every time the pipeline decides CREATE_NEW_PATCH (i.e. no domain expert is
available for the query), this module appends a record to a per-domain JSONL
file and later patches the record with the final response.

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


def _sanitize_tag(value: str) -> str:
    """Return a filesystem-safe lowercase tag token."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value.lower())
    safe = safe.strip("_")
    return safe or "unclassified"


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
    - ``log_query()`` writes the record to today's per-domain JSONL file immediately.
    - ``_rewrite_record()`` patches the ``response`` field in-place when
      ``fill_response()`` is called.
    """

    def __init__(self, batch_dir: Path = _BATCH_DIR) -> None:
        self._batch_dir = batch_dir
        self._batch_dir.mkdir(parents=True, exist_ok=True)
        self._pending: Dict[str, Dict[str, Any]] = {}
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
        """Append a CREATE_NEW_PATCH record and mark it pending."""
        _metadata = metadata or {}
        domain_tag = _sanitize_tag(
            _metadata.get("domain_tag")
            or (domains[0] if domains else None)
            or (tags[0] if tags else None)
            or "unclassified"
        )
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
            "metadata": _metadata,
        }
        path = self._today_path(domain_tag)
        with self._lock:
            self._pending[trace_id] = {"record": record, "path": path}
        self._append(record, path)

    def fill_response(
        self,
        trace_id: str,
        response: str,
        *,
        sandbox_evidence: Optional[List[Dict[str, Any]]] = None,
        phase_latencies_ms: Optional[Dict[str, float]] = None,
    ) -> None:
        """Patch the response field of a previously logged record."""
        with self._lock:
            pending = self._pending.get(trace_id)
            if pending is None:
                return
            record = pending["record"]
            path = pending["path"]
            record["response"] = response
            if sandbox_evidence is not None:
                record["sandbox_evidence"] = sandbox_evidence
            if phase_latencies_ms is not None:
                record["phase_latencies_ms"] = phase_latencies_ms
            self._pending.pop(trace_id, None)
        self._rewrite_record(trace_id, record, path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _today_path(self, domain_tag: str = "unclassified") -> Path:
        date_str = datetime.now(timezone.utc).strftime(_DATE_FMT)
        p = self._batch_dir / domain_tag / f"{date_str}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _append(self, record: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _rewrite_record(
        self, trace_id: str, updated: Dict[str, Any], path: Path
    ) -> None:
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
                new_lines.append(json.dumps(updated, ensure_ascii=False) + "\n")
            else:
                new_lines.append(line)
        path.write_text("".join(new_lines), encoding="utf-8")

    def _flush_all(self) -> None:
        """Flush any still-pending (no-response) records at exit.

        Called automatically on process exit via ``atexit``.  Pending records
        are written with an empty ``response`` field so they are not lost.
        """
        with self._lock:
            orphans = list(self._pending.values())
            self._pending.clear()

        if not orphans:
            return

        for entry in orphans:
            record = entry["record"]
            path = entry["path"]
            try:
                self._append(record, path)
            except Exception as exc:
                print(
                    f"⚠️  PatchBatchLogger: failed to flush pending record {record.get('trace_id')}: {exc}"
                )


# Process-wide singleton
patch_logger = PatchBatchLogger()
