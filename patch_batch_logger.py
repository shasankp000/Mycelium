"""
patch_batch_logger.py

Thread-safe logger for CREATE_NEW_PATCH events.

Every time the pipeline decides CREATE_NEW_PATCH (i.e. no domain expert is
available for the query), this module:

  1. Immediately appends a JSONL record containing the query + metadata to
     ``patch_dataset/<domain_tag>/<YYYY-MM-DD>.jsonl``.
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
        "routing": {
            "layer0_route": str,
            "classification": str,
            "domains": list,
            "expert_decision": str,
            "confidence": float
        },
        "response":               str,   # filled after LLM generation
        "sandbox_evidence":       list,
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

_BATCH_DIR = Path(
    os.getenv("PATCH_DATASET_DIR", os.getenv("PATCH_BATCH_DIR", "patch_dataset"))
)
_DATE_FMT = "%Y-%m-%d"


def _sanitize_tag(value: str) -> str:
    """Return a filesystem-safe lowercase tag token."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value.lower())
    safe = safe.strip("_")
    return safe or "unclassified"


class PatchBatchLogger:
    """Append-only JSONL logger for CREATE_NEW_PATCH events.

    A single process-wide instance is created at module level and exported
    as ``patch_logger``.  Import and use it directly::

        from patch_batch_logger import patch_logger

        patch_logger.log_query(
            trace_id=..., query=..., tags=...,
            layer0_route="REASONING_PIPELINE",
            routing_classification=...,
            domains=..., expert_decision="CREATE_NEW_PATCH",
            confidence=0.0, phase_latencies_ms=...
        )
        # … pipeline runs …
        patch_logger.fill_response(trace_id=..., response=final_answer)
    """

    def __init__(self, batch_dir: Path = _BATCH_DIR) -> None:
        self._batch_dir = batch_dir
        self._batch_dir.mkdir(parents=True, exist_ok=True)
        # In-flight records keyed by trace_id (awaiting response)
        # value contains: {"record": Dict[str, Any], "path": Path}
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
        layer0_route: Optional[str] = None,
        routing_classification: Optional[str] = None,
        domains: Optional[List[str]] = None,
        expert_decision: Optional[str] = None,
        confidence: Optional[float] = None,
        phase_latencies_ms: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a CREATE_NEW_PATCH event immediately.

        The ``response`` field is left empty and filled later by
        :meth:`fill_response`.
        """
        metadata = metadata or {}
        domains = domains or []
        tags = tags or []
        domain_tag = _sanitize_tag(
            str(
                metadata.get("domain_tag")
                or (domains[0] if domains else "")
                or (tags[0] if tags else "")
                or "unclassified"
            )
        )
        record: Dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "routing": {
                "layer0_route": layer0_route or "REASONING_PIPELINE",
                "classification": routing_classification or "",
                "domains": domains,
                "expert_decision": expert_decision or "CREATE_NEW_PATCH",
                "confidence": float(confidence or 0.0),
            },
            "tags": tags,
            "sandbox_evidence": [],
            "response": "",
            "phase_latencies_ms": phase_latencies_ms or {},
            "metadata": metadata,
        }
        with self._lock:
            path = self._today_path(domain_tag)
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
        """Patch the ``response`` field of a previously logged record.

        Rewrites the daily JSONL file with the updated record in-place.  For
        the typical single-writer, low-volume case this is acceptable; a
        database backend can replace this if volume grows.
        """
        with self._lock:
            pending = self._pending.get(trace_id)
            if pending is None:
                # Already flushed or unknown trace — nothing to do.
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
        domain_dir = self._batch_dir / domain_tag
        domain_dir.mkdir(parents=True, exist_ok=True)
        return domain_dir / f"{date_str}.jsonl"

    def _append(self, record: Dict[str, Any], path: Path) -> None:
        """Append a single JSON line to today's batch file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _rewrite_record(
        self, trace_id: str, updated: Dict[str, Any], path: Path
    ) -> None:
        """Scan the daily file and replace the line matching trace_id."""
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
