"""
sandbox_manager.py  (Milestone 3 — MCP + DomainToolPlanner revision)
======================================================================
LLM-free sandbox orchestration.

See original docstring for full architecture notes.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from mycelium.pipeline.domain_tool_planner import DomainToolPlanner, ToolCall, get_planner
from mycelium.pipeline.mcp_client import MCPClient, get_mcp_client
from mycelium.pipeline.sandbox_models import SandboxResult, SandboxStep, SandboxTask

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STEPS = 4
_DEFAULT_WALL_TIMEOUT_S = 30.0

ProgressCallback = Callable[[str, str], None]


def _extract_preview(tool: str, output: Dict[str, Any], max_len: int = 100) -> str:
    if not output:
        return ""
    if output.get("status") == "error":
        err = output.get("error") or output.get("message") or "unknown error"
        return str(err)[:max_len]
    results = output.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        snippet = first.get("snippet") or first.get("abstract") or first.get("title") or ""
        return str(snippet)[:max_len]
    papers = output.get("papers")
    if isinstance(papers, list) and papers:
        first = papers[0]
        title = first.get("title") or ""
        year  = first.get("year") or ""
        preview = f"{title} ({year})" if year else title
        return str(preview)[:max_len]
    entities = output.get("entities")
    if isinstance(entities, list) and entities:
        first = entities[0]
        label = first.get("label") or ""
        desc  = first.get("description") or ""
        preview = f"{label}: {desc}" if desc else label
        return str(preview)[:max_len]
    calc_result = output.get("result")
    if calc_result is not None:
        return str(calc_result)[:max_len]
    for v in output.values():
        if isinstance(v, str) and v.strip():
            return v.strip()[:max_len]
    return ""


class SandboxManager:
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

    def run(
        self,
        task: SandboxTask,
        on_progress: Optional[ProgressCallback] = None,
    ) -> SandboxResult:
        def _emit(phase: str, detail: str) -> None:
            if on_progress:
                try:
                    on_progress(phase, detail)
                except Exception:
                    pass
            logger.info("[sandbox] phase=%-28s  %s", phase, detail)

        wall_start = time.monotonic()
        started = datetime.utcnow()
        steps: List[SandboxStep] = []

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
        _emit("sandbox_plan", f"{len(plan)} | {', '.join(tool_names)}")
        logger.info(
            "SandboxManager: plan for trace=%s domains=%s → tools=%s",
            task.trace_id, task.domains, tool_names,
        )

        for idx, call in enumerate(plan[: self._max_steps], start=1):
            elapsed_s = time.monotonic() - wall_start
            if elapsed_s > self._wall_timeout_s * 0.85:
                warn = f"Wall timeout approaching ({elapsed_s:.1f}s / {self._wall_timeout_s}s), stopping early"
                logger.warning("SandboxManager: %s", warn)
                _emit(f"sandbox_tool/{idx}_err", f"{call.tool} | ✗ 0ms | {warn}")
                break

            _emit(f"sandbox_tool/{idx}", f"{call.tool} | {call.query[:80]}")

            step_wall = time.monotonic()
            step = self._execute_step(call)
            duration_ms = round((time.monotonic() - step_wall) * 1000)
            steps.append(step)

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
        return SandboxResult(
            trace_id=task.trace_id,
            started_at=started,
            finished_at=finished,
            steps=steps,
            summary="",
        )

    def _execute_step(self, call: ToolCall) -> SandboxStep:
        step_start = datetime.utcnow()
        if call.tool == "calculator":
            arguments: Dict[str, Any] = {"expression": call.query}
        else:
            arguments = {"query": call.query}
        logger.debug("SandboxManager._execute_step: tool=%s arguments=%s", call.tool, arguments)
        try:
            output = self._mcp.call_tool(call.tool, arguments)
            status = "error" if output.get("status") == "error" else "ok"
            logger.debug(
                "SandboxManager._execute_step: tool=%s status=%s output_keys=%s",
                call.tool, status, list(output.keys())[:6],
            )
        except Exception as exc:
            output = {"error": str(exc), "status": "error", "source": call.tool}
            status = "error"
            logger.warning("SandboxManager: MCP call '%s' raised — %s", call.tool, exc)
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
    def _stub_result(task: SandboxTask, started: datetime, reason: str = "") -> SandboxResult:
        return SandboxResult(
            trace_id=task.trace_id,
            started_at=started,
            finished_at=datetime.utcnow(),
            steps=[],
            summary=(f"Sandbox produced no results. {reason}".strip()),
        )


_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    return _manager
