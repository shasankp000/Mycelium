"""
sandbox_manager.py  (Milestone 3 — MCP + DomainToolPlanner revision)
======================================================================
LLM-free sandbox orchestration.

Architecture
------------

    broadcast_api
        └─ SandboxManager.run(task, on_progress=...)
                ├─ DomainToolPlanner.plan()   ← deterministic, no LLM
                │       (TRM seam: swap for TRM.plan_tools() when ready)
                └─ MCPClient.call_tool()      ← MCP stdio transport
                        (mcp_tools_server.py subprocess)

The LLM is NO LONGER called inside SandboxManager.  All synthesis
(evidence summary + plain-language explanation) is done in a single
Ollama round-trip inside ConversationAgent.answer().

This means:
  - Zero cold-starts inside the sandbox path.
  - ConversationAgent receives the raw SandboxResult and does both jobs.
  - Total LLM calls per request: exactly 1 (down from 2).

on_progress callback
--------------------
An optional callable(phase: str, detail: str) is accepted by run().
It is called synchronously at every meaningful sub-step so callers
(broadcast_api SSE generator) can forward progress to the frontend.

Phases emitted from inside SandboxManager:
  sandbox_plan        — tool plan produced, tools listed
  sandbox_tool/<n>    — MCP call n started  (detail: "web_search | query text")
  sandbox_tool/<n>_ok — MCP call n succeeded (detail: "web_search | ✓ 423ms | preview…")
  sandbox_tool/<n>_err— MCP call n failed    (detail: "web_search | ✗ 423ms | error msg")

Detail format (pipe-separated so frontend can parse without fragile regexes):
  START  : "<tool> | <query[:80]>"
  OK     : "<tool> | ✓ <duration_ms>ms | <output_preview[:100]>"
  ERR    : "<tool> | ✗ <duration_ms>ms | <error_message[:100]>"

Changes (2026-05-16 — patch 5)
------------------------------
  - on_progress detail strings for _ok/_err now include a short output
    preview extracted from the MCP response so the frontend can display
    it in-place without waiting for the final `done` payload.
  - START detail now uses pipe-separated format: "<tool> | <query>"
  - _extract_preview() helper added for structured output parsing.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from domain_tool_planner import DomainToolPlanner, ToolCall, get_planner
from mcp_client import MCPClient, get_mcp_client
from sandbox_models import SandboxResult, SandboxStep, SandboxTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------
_DEFAULT_MAX_STEPS = 4
_DEFAULT_WALL_TIMEOUT_S = 30.0

# Type alias for the progress callback
ProgressCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Output preview extractor
# ---------------------------------------------------------------------------

def _extract_preview(tool: str, output: Dict[str, Any], max_len: int = 100) -> str:
    """
    Extract a short human-readable preview from an MCP tool output dict.

    Supports the structured {status, source, results/papers/entities/result}
    shapes emitted by mcp_tools_server's four built-in tools.

    Returns an empty string when nothing useful can be extracted.
    """
    if not output:
        return ""

    # Error case
    if output.get("status") == "error":
        err = output.get("error") or output.get("message") or "unknown error"
        return str(err)[:max_len]

    # web_search / general search → list of {title, snippet, url}
    results = output.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        snippet = first.get("snippet") or first.get("abstract") or first.get("title") or ""
        return str(snippet)[:max_len]

    # academic_search → list of {title, authors, year, abstract, url}
    papers = output.get("papers")
    if isinstance(papers, list) and papers:
        first = papers[0]
        title = first.get("title") or ""
        year  = first.get("year") or ""
        preview = f"{title} ({year})" if year else title
        return str(preview)[:max_len]

    # knowledge_base → list of {label, description, url}
    entities = output.get("entities")
    if isinstance(entities, list) and entities:
        first = entities[0]
        label = first.get("label") or ""
        desc  = first.get("description") or ""
        preview = f"{label}: {desc}" if desc else label
        return str(preview)[:max_len]

    # calculator → {result: <number|string>}
    calc_result = output.get("result")
    if calc_result is not None:
        return str(calc_result)[:max_len]

    # Generic fallback — first string value found
    for v in output.values():
        if isinstance(v, str) and v.strip():
            return v.strip()[:max_len]

    return ""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SandboxManager:
    """
    Orchestrates a sandbox run for a single Mycelium pipeline execution.

    Planning is fully deterministic (DomainToolPlanner).  Execution goes
    through the MCP stdio client (MCPClient → mcp_tools_server subprocess).

    The LLM is NOT invoked here.  SandboxResult.summary is always ""
    (empty string).  ConversationAgent.answer() performs both evidence
    synthesis and plain-language explanation in a single LLM call.
    """

    def __init__(
        self,
        planner: Optional[DomainToolPlanner] = None,
        mcp: Optional[MCPClient] = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
        wall_timeout_s: float = _DEFAULT_WALL_TIMEOUT_S,
    ) -> None:
        self._planner = planner or get_planner()
        self._mcp = mcp or get_mcp_client()
        self._max_steps = max_steps
        self._wall_timeout_s = wall_timeout_s

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        task: SandboxTask,
        on_progress: Optional[ProgressCallback] = None,
    ) -> SandboxResult:
        """
        Execute the full sandbox pipeline for *task*.

        Parameters
        ----------
        task : SandboxTask
        on_progress : optional callable(phase, detail)
            Called synchronously at each sub-step so the SSE generator in
            broadcast_api can forward live progress events to the frontend.

            Detail format (pipe-separated):
              START  : "<tool> | <query[:80]>"
              OK     : "<tool> | ✓ <duration_ms>ms | <output_preview[:100]>"
              ERR    : "<tool> | ✗ <duration_ms>ms | <error[:100]>"

        Steps
        -----
        1. DomainToolPlanner produces a minimal ToolCallPlan (no LLM).
        2. Each ToolCall is dispatched via MCPClient (MCP stdio transport).
        3. SandboxResult returned with summary="".
           ConversationAgent.answer() handles LLM synthesis downstream.

        Never raises — all exceptions are caught and returned as error
        SandboxStep records.
        """
        def _emit(phase: str, detail: str) -> None:
            if on_progress:
                try:
                    on_progress(phase, detail)
                except Exception:
                    pass  # never let a progress callback crash the sandbox
            logger.info("[sandbox] phase=%-28s  %s", phase, detail)

        wall_start = time.monotonic()
        started = datetime.utcnow()
        steps: List[SandboxStep] = []

        # ── Step 1: deterministic planning ──────────────────────────────
        decision_type = task.decision_type if hasattr(task, "decision_type") else ""
        plan: List[ToolCall] = self._planner.plan(
            query=task.user_query,
            domains=list(task.domains),
            decision_type=decision_type,
        )

        if not plan:
            _emit("sandbox_plan", "DomainToolPlanner returned empty plan")
            return self._stub_result(task, started, reason="DomainToolPlanner returned empty plan")

        tool_names = [c.tool for c in plan]
        _emit(
            "sandbox_plan",
            f"{len(plan)} | {', '.join(tool_names)}",
        )
        logger.info(
            "SandboxManager: plan for trace=%s domains=%s → tools=%s",
            task.trace_id, task.domains, tool_names,
        )

        # ── Step 2: MCP execution ────────────────────────────────────────
        for idx, call in enumerate(plan[: self._max_steps], start=1):
            elapsed_s = time.monotonic() - wall_start
            if elapsed_s > self._wall_timeout_s * 0.85:
                warn = f"Wall timeout approaching ({elapsed_s:.1f}s / {self._wall_timeout_s}s), stopping early"
                logger.warning("SandboxManager: %s", warn)
                # emit timeout as an error row so frontend can show it
                _emit(
                    f"sandbox_tool/{idx}_err",
                    f"{call.tool} | ✗ 0ms | {warn}",
                )
                break

            # START event — pipe-separated so frontend can parse without regex
            _emit(
                f"sandbox_tool/{idx}",
                f"{call.tool} | {call.query[:80]}",
            )

            step_wall = time.monotonic()
            step = self._execute_step(call)
            duration_ms = round((time.monotonic() - step_wall) * 1000)

            steps.append(step)

            # RESULT event — include short output preview
            preview = _extract_preview(call.tool, step.output)
            if step.status == "ok":
                result_detail = f"{call.tool} | ✓ {duration_ms}ms"
                if preview:
                    result_detail += f" | {preview}"
            else:
                result_detail = f"{call.tool} | ✗ {duration_ms}ms"
                if preview:
                    result_detail += f" | {preview}"

            status_phase = f"sandbox_tool/{idx}_{'ok' if step.status == 'ok' else 'err'}"
            _emit(status_phase, result_detail)

        finished = datetime.utcnow()

        # summary is intentionally empty — ConversationAgent does synthesis
        return SandboxResult(
            trace_id=task.trace_id,
            started_at=started,
            finished_at=finished,
            steps=steps,
            summary="",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_step(self, call: ToolCall) -> SandboxStep:
        """Dispatch a single ToolCall via MCP and record the result."""
        step_start = datetime.utcnow()

        if call.tool == "calculator":
            arguments: Dict[str, Any] = {"expression": call.query}
        else:
            arguments = {"query": call.query}

        logger.debug(
            "SandboxManager._execute_step: tool=%s arguments=%s",
            call.tool, arguments,
        )

        try:
            output = self._mcp.call_tool(call.tool, arguments)
            # Use output.status field for reliable error detection.
            # Avoid `"error" in output` (dict key check) which fires on any
            # output dict that happens to contain an "error" key, even when
            # the call succeeded (e.g. web_search returning a "note" key
            # alongside valid results).
            status = "error" if output.get("status") == "error" else "ok"
            logger.debug(
                "SandboxManager._execute_step: tool=%s status=%s output_keys=%s",
                call.tool, status, list(output.keys())[:6],
            )
        except Exception as exc:
            output = {"error": str(exc), "status": "error", "source": call.tool}
            status = "error"
            logger.warning(
                "SandboxManager: MCP call '%s' raised — %s", call.tool, exc
            )

        step_end = datetime.utcnow()
        return SandboxStep(
            tool=call.tool,
            input={"query": call.query},
            output=output,
            commentary=f"Planned by DomainToolPlanner for tool: {call.tool}",
            status=status,
            started_at=step_start,
            finished_at=step_end,
        )

    @staticmethod
    def _stub_result(
        task: SandboxTask, started: datetime, reason: str = ""
    ) -> SandboxResult:
        return SandboxResult(
            trace_id=task.trace_id,
            started_at=started,
            finished_at=datetime.utcnow(),
            steps=[],
            summary=(
                f"Sandbox produced no results. {reason}".strip()
            ),
        )


# Process-level singleton
_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    return _manager
