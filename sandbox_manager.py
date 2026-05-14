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
  sandbox_tool/<n>    — MCP call n started
  sandbox_tool/<n>_ok — MCP call n succeeded
  sandbox_tool/<n>_err— MCP call n failed
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
        # Note: llm parameter removed — SandboxManager no longer owns an LLMClient.
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
