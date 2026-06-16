"""
mycelium/pipeline/run_workflow.py — Legacy pipeline shim.

This module re-exposes ``run_mycelium_workflow`` from the archived TRM v1
pipeline under the original import path so that the root
``run_workflow.run_legacy()`` can continue to delegate to it.

DO NOT add new logic here.  This shim exists only to bridge the legacy call
path while TRM v2 reaches full parity.  Once parity is confirmed, delete
both this file and ``run_legacy()`` in the root run_workflow.py together.

See: archive/legacy_run_workflow_v1.py for the full archived pipeline.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence

warnings.warn(
    "mycelium.pipeline.run_workflow is the legacy TRM v1 pipeline shim. "
    "Use --enable-trm-v2 to run the TRM v2 path. "
    "This shim will be removed once TRM v2 reaches parity.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the archived v1 entry-point.  Hard failure is intentional: if the
# archive directory is missing the operator needs to know immediately rather
# than silently running nothing.
# ---------------------------------------------------------------------------
try:
    from archive.legacy_run_workflow_v1 import (  # type: ignore[import]
        run_mycelium_workflow as _run_mycelium_workflow,
        WorkflowMetrics as _WorkflowMetrics,
    )
    _LEGACY_AVAILABLE = True
except ImportError as _exc:
    logger.warning(
        "Legacy archive not importable (%s). "
        "Legacy mode will be unavailable — use --enable-trm-v2.",
        _exc,
    )
    _run_mycelium_workflow = None  # type: ignore[assignment]
    _WorkflowMetrics = None  # type: ignore[assignment]
    _LEGACY_AVAILABLE = False


def run(
    query: Optional[str] = None,
    backend_only: bool = False,
    sentences: Optional[Sequence[str]] = None,
    reasoning_mode: str = "smart",
    output_path: Optional[str] = None,
) -> None:
    """
    Adapter: accepts the ``(query, backend_only)`` call shape used by the
    root ``run_workflow.run_legacy()``, converts it to the positional
    ``run_mycelium_workflow(sentences, ...)`` signature expected by the
    archived v1 pipeline, and streams / prints results.

    Parameters
    ----------
    query:
        Single query string.  If *sentences* is also provided, *query* is
        prepended to that list.
    backend_only:
        When True, print JSON result and exit.  When False, drop into the
        legacy REPL (reads from stdin).
    sentences:
        Explicit list of sentences to process.  Overrides REPL mode.
    reasoning_mode:
        Forwarded verbatim to ``run_mycelium_workflow``.  One of:
        ``"fast"`` | ``"smart"`` | ``"deep"``.
    output_path:
        Optional path to write the JSON result.  If None, prints to stdout.
    """
    if not _LEGACY_AVAILABLE or _run_mycelium_workflow is None:
        raise RuntimeError(
            "Legacy pipeline is not available (archive import failed). "
            "Run with --enable-trm-v2 instead."
        )

    # --- Build the sentences list ------------------------------------------
    if sentences is not None:
        _sentences: List[str] = list(sentences)
        if query:
            _sentences.insert(0, query)
    elif backend_only:
        if not query:
            print(
                "ERROR: --query is required when --backend-only is set (legacy mode).",
                file=sys.stderr,
            )
            sys.exit(1)
        _sentences = [query]
    else:
        # Interactive REPL: collect sentences from stdin
        _sentences = _legacy_repl()
        if not _sentences:
            return

    # --- Run the archived v1 pipeline --------------------------------------
    collected_events: List[Dict[str, Any]] = []

    def _on_event(ev: Dict[str, Any]) -> None:
        collected_events.append(ev)

    results, metrics = _run_mycelium_workflow(
        sentences=_sentences,
        on_event=_on_event,
        reasoning_mode=reasoning_mode,
    )

    # --- Serialise output --------------------------------------------------
    output: Dict[str, Any] = {
        "results": results,
        "metrics": metrics.to_dict() if metrics is not None else {},
    }

    _emit_output(output, output_path)


def _legacy_repl() -> List[str]:
    """Read sentences interactively from stdin until EOF or 'quit'."""
    sentences: List[str] = []
    print("Mycelium legacy mode — enter sentences (empty line / 'quit' to run).\n")
    try:
        while True:
            try:
                line = input("sentence> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.lower() in ("quit", "exit", "q", ""):
                break
            sentences.append(line)
    except Exception as exc:
        logger.warning("REPL read error: %s", exc)
    return sentences


def _emit_output(output: Dict[str, Any], output_path: Optional[str]) -> None:
    """Print JSON to stdout or write to *output_path*."""
    import numpy as np  # local import — may not be available in all envs

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            return super().default(o)

    payload = json.dumps(output, indent=2, cls=_NumpyEncoder)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"Legacy output written to {output_path}")
    else:
        print(payload)
