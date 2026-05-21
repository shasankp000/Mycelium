"""
patch_dataset_logger.py  —  Milestone 1.5

Writes and updates patch-dataset JSONL records whenever Mycelium encounters
a CREATE_NEW_PATCH routing decision (i.e. no domain expert exists).

Directory layout:
    patch_dataset/
        <domain_tag>/
            <YYYY-MM-DD>.jsonl    ← one JSON object per line, newline-delimited

Each line is a PatchRecord (see api_models.PatchRecord).  Records are written
in two passes:

  Pass 1  — call log_patch_query() BEFORE the pipeline runs.
             Writes a record with final_answer=null.

  Pass 2  — call update_with_answer() AFTER the pipeline completes.
             Appends an updated record (or patches the most recent matching
             trace_id) with the resolved answer.

Thread safety: a per-file threading.Lock ensures concurrent FastAPI coroutines
(running in the same process via asyncio + a threadpool) don’t interleave
writes to the same file.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from mycelium.pipeline.api_models import PatchRecord

# Root directory relative to the repo root (wherever broadcast_api.py lives)
PATCH_DATASET_ROOT = Path("patch_dataset")

# Global per-file lock registry
_file_locks: dict[Path, threading.Lock] = {}
_registry_lock = threading.Lock()


def _get_file_lock(path: Path) -> threading.Lock:
    with _registry_lock:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]


class PatchDatasetLogger:
    """
    Instantiated once per request (or once at startup and reused — both
    patterns are safe because all state is written to disk, not held in
    instance variables across requests).
    """

    def _resolve_path(self, domain_tag: str) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_dir = PATCH_DATASET_ROOT / domain_tag
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{today}.jsonl"

    # ------------------------------------------------------------------
    # Pass 1  — write stub record before pipeline runs
    # ------------------------------------------------------------------

    def log_patch_query(
        self,
        trace_id: str,
        domain_tag: str,
        user_query: str,
        routing_domains: Optional[List[str]] = None,
        confidence_at_routing: Optional[float] = None,
    ) -> Path:
        """
        Write a PatchRecord stub with final_answer=None.
        Returns the path that was written to (for inclusion in ChatResponse).
        """
        record = PatchRecord(
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            domain_tag=domain_tag,
            user_query=user_query,
            routing_domains=routing_domains or [],
            final_answer=None,
            confidence_at_routing=confidence_at_routing,
        )
        path = self._resolve_path(domain_tag)
        lock = _get_file_lock(path)
        with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")
        return path

    # ------------------------------------------------------------------
    # Pass 2  — update with final answer after pipeline completes
    # ------------------------------------------------------------------

    def update_with_answer(
        self,
        trace_id: str,
        domain_tag: str,
        final_answer: str,
        sandbox_summary: Optional[str] = None,
    ) -> None:
        """
        Appends an updated copy of the record with the resolved answer.
        Rather than rewriting the whole file (which is expensive on large
        datasets), we simply append a second record with the same trace_id.
        During training, the last record for a given trace_id wins.
        """
        path = self._resolve_path(domain_tag)
        if not path.exists():
            # Edge case: the stub was never written (e.g. log_patch_query
            # was skipped). Write a standalone record.
            self.log_patch_query(
                trace_id=trace_id,
                domain_tag=domain_tag,
                user_query="<unknown>",
            )

        record = PatchRecord(
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            domain_tag=domain_tag,
            user_query="<see stub record>",
            final_answer=final_answer,
            sandbox_summary=sandbox_summary,
        )
        lock = _get_file_lock(path)
        with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # Stats helper  (used by GET /api/v1/patch/stats)
    # ------------------------------------------------------------------

    @staticmethod
    def get_stats() -> dict:
        """
        Returns {domain_tag: count_of_records} across all JSONL files.
        Only counts stub records (final_answer == null) to reflect
        unhandled queries, not duplicates.
        """
        stats: dict[str, int] = {}
        if not PATCH_DATASET_ROOT.exists():
            return stats
        for domain_dir in PATCH_DATASET_ROOT.iterdir():
            if not domain_dir.is_dir():
                continue
            count = 0
            for jsonl_file in domain_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                # Only count stub entries (no final_answer)
                                if obj.get("final_answer") is None:
                                    count += 1
                            except json.JSONDecodeError:
                                pass
                except OSError:
                    pass
            stats[domain_dir.name] = count
        return stats
