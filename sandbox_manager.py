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
                ├─ MCPClient.call_tool()      ← MCP stdio transport
                │       (mcp_tools_server.py subprocess)
                └─ LLMClient.generate()       ← evidence summary ONLY
                        (ConversationAgent handles user-facing language)

The LLM is invoked exactly once per sandbox run — to write a 3-5 sentence
evidence summary from the raw tool results.  It is never asked to decide
which tools to call; that is the exclusive responsibility of DomainToolPlanner
(and eventually the real TRM).

on_progress callback
--------------------
An optional callable(phase: str, detail: str) is accepted by run().
It is called synchronously at every meaningful sub-step so callers
(broadcast_api SSE generator) can forward progress to the frontend.

Phases emitted from inside SandboxManager:
  sandbox_plan        — tool plan produced, tools listed
  sandbox_tool/<n>    — MCP call n started
  sandbox_tool/<n>_ok — MCP call n succeeded
  sandbox_tool/<n>_err— MCP call n failed
  sandbox_summary_start — LLM summary call starting (this is the slow step)
  sandbox_summary_done  — LLM summary call complete
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from domain_tool_planner import DomainToolPlanner, ToolCall, get_planner
from llm_providers import LLMClient
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
# LLM prompt
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = """\
You are the Mycelium sandbox summariser. Given a user query and a list of
tool results, write a concise evidence summary (3-5 sentences). Rules:
- Only reference facts actually present in the tool results.
- Note source names (e.g. "Semantic Scholar", "Wikidata", "DuckDuckGo").
- Flag if evidence is sparse or contradictory.
- Do not invent facts.
- Do not suggest further tool calls.
"""

_SUMMARY_USER_TMPL = """\
User query: {query}

Tool results:
{results_block}

Write the evidence summary now.
"""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SandboxManager:
    """
    Orchestrates a sandbox run for a single Mycelium pipeline execution.

    Planning is fully deterministic (DomainToolPlanner).  Execution goes
    through the MCP stdio client (MCPClient → mcp_tools_server subprocess).
    The LLM is used only to summarise collected evidence.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        planner: Optional[DomainToolPlanner] = None,
        mcp: Optional[MCPClient] = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
        wall_timeout_s: float = _DEFAULT_WALL_TIMEOUT_S,
    ) -> None:
        self._llm = llm or LLMClient()
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

        Steps
        -----
        1. DomainToolPlanner produces a minimal ToolCallPlan (no LLM).
        2. Each ToolCall is dispatched via MCPClient (MCP stdio transport).
        3. LLM writes a 3-5 sentence evidence summary from raw results.

        Never raises — all exceptions are caught and returned as error
        SandboxStep records or an error summary string.
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
            f"Plan ready — {len(plan)} tool(s): {', '.join(tool_names)}",
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
                _emit(f"sandbox_tool/{idx}", f"⚠ {warn}")
                break

            _emit(
                f"sandbox_tool/{idx}",
                f"Calling {call.tool} — query: {call.query[:80]}",
            )
            step = self._execute_step(call)
            steps.append(step)

            duration_ms = (
                (step.finished_at - step.started_at).total_seconds() * 1000
                if step.finished_at and step.started_at else 0
            )
            status_phase = f"sandbox_tool/{idx}_{'ok' if step.status == 'ok' else 'err'}"
            _emit(
                status_phase,
                f"{call.tool} {'✓' if step.status == 'ok' else '✗'} in {duration_ms:.0f}ms",
            )

        # ── Step 3: LLM evidence summary ────────────────────────────────
        _emit(
            "sandbox_summary_start",
            f"Asking LLM to summarise {len([s for s in steps if s.status == 'ok'])} evidence result(s)…",
        )
        summary = self._synthesise_summary(task.user_query, steps)
        _emit("sandbox_summary_done", "Evidence summary ready")
        finished = datetime.utcnow()

        return SandboxResult(
            trace_id=task.trace_id,
            started_at=started,
            finished_at=finished,
            steps=steps,
            summary=summary,
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
            status = "error" if "error" in output else "ok"
            logger.debug(
                "SandboxManager._execute_step: tool=%s status=%s output_keys=%s",
                call.tool, status, list(output.keys())[:6],
            )
        except Exception as exc:
            output = {"error": str(exc)}
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

    def _synthesise_summary(self, query: str, steps: List[SandboxStep]) -> str:
        """Ask the LLM to write a 3-5 sentence evidence summary."""
        if not steps:
            return "No tool calls were executed; no evidence was gathered."

        ok_steps = [s for s in steps if s.status == "ok"]
        if not ok_steps:
            return (
                f"All {len(steps)} tool call(s) failed. "
                "No external evidence could be retrieved for this query."
            )

        results_lines: List[str] = []
        for i, s in enumerate(ok_steps, 1):
            out_str = json.dumps(s.output, ensure_ascii=False)[:600]
            results_lines.append(
                f"[{i}] tool={s.tool} input={s.input.get('query', '')}\n"
                f"    output={out_str}"
            )
        results_block = "\n\n".join(results_lines)

        user_msg = _SUMMARY_USER_TMPL.format(
            query=query, results_block=results_block
        )
        logger.info(
            "SandboxManager._synthesise_summary: calling LLM for %d ok step(s)",
            len(ok_steps),
        )
        t0 = time.monotonic()
        try:
            result = self._llm.generate(user_msg, system=_SUMMARY_SYSTEM)
            logger.info(
                "SandboxManager._synthesise_summary: LLM done in %.1fs",
                time.monotonic() - t0,
            )
            return result
        except Exception as exc:
            logger.warning(
                "SandboxManager._synthesise_summary: LLM call failed in %.1fs — %s",
                time.monotonic() - t0, exc,
            )
            return (
                f"Sandbox ran {len(steps)} tool call(s), "
                f"{len(ok_steps)} succeeded. "
                "LLM summary unavailable — see individual steps for raw evidence."
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
