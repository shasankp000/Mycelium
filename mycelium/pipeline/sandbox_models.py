"""sandbox_models.py  (Milestone 3)

Pydantic data models for the Mycelium sandbox layer.

All models use Pydantic v2 BaseModel and are fully JSON-serialisable so
they can be embedded inside ReasoningTrace records on disk and returned
directly from the FastAPI /api/v1/chat endpoint.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Task — what the sandbox receives
# ---------------------------------------------------------------------------

class SandboxTask(BaseModel):
    """Describes a single sandbox run request."""

    trace_id: str
    user_query: str

    # Domains inferred from the routing layer (e.g. ["physics", "mathematics"])
    domains: List[str] = Field(default_factory=list)

    # Hypotheses / intermediate conclusions from Phase 3-5 that the sandbox
    # should attempt to verify or extend.
    hypotheses: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step — one tool call executed by the sandbox
# ---------------------------------------------------------------------------

class SandboxStep(BaseModel):
    """Records a single tool invocation inside a sandbox run."""

    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    commentary: str = ""
    status: str = "ok"          # "ok" | "error"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Result — full outcome of a sandbox run
# ---------------------------------------------------------------------------

class SandboxResult(BaseModel):
    """Aggregated result returned by SandboxManager.run()."""

    trace_id: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    steps: List[SandboxStep] = Field(default_factory=list)

    # Plain-language summary produced by the LLM from all tool results.
    summary: str = ""


# ---------------------------------------------------------------------------
# Helper: build a SandboxTask from a completed pipeline run
# ---------------------------------------------------------------------------

def build_sandbox_task_from_run(
    run: Any,
    user_query: str = "",
) -> SandboxTask:
    """Derive a SandboxTask from a MyceliumRunSummary (or compatible object/dict).

    Extracts domains and any intermediate hypotheses that Phase 3-5 may have
    recorded so the sandbox can target its tool calls precisely.

    Parameters
    ----------
    run:
        A MyceliumRunSummary instance, a Pydantic model, or a plain dict.
    user_query:
        The original user question.  Falls back to run.sentence if omitted.
    """
    try:
        d: Dict[str, Any] = run.model_dump()  # type: ignore[attr-defined]
    except AttributeError:
        d = dict(run) if not isinstance(run, dict) else run  # type: ignore

    # --- trace id ---
    trace_id: str = d.get("trace_id") or ""

    # --- user query ---
    if not user_query:
        user_query = d.get("sentence") or ""

    # --- domains ---
    routing: Dict[str, Any] = d.get("routing") or {}
    domains: List[str] = list(routing.get("selected_domains") or [])

    # --- hypotheses from phase3 action_result / validated answer ---
    hypotheses: List[str] = []
    phase3: Dict[str, Any] = d.get("phase3") or {}
    action_result: Dict[str, Any] = phase3.get("action_result") or {}

    # Try common keys where Phase 3-5 might store a partial answer
    for key in ("validated_answer", "verified_answer", "action_answer",
                "result", "answer"):
        val = action_result.get(key)
        if val and isinstance(val, str) and len(val) > 10:
            hypotheses.append(val[:400])
            break

    # Phase3 raw block — look for consequence strings
    raw_phase3: Dict[str, Any] = phase3.get("raw") or {}
    for key in ("hypotheses", "consequences", "evidence_chains"):
        items = raw_phase3.get(key)
        if isinstance(items, list):
            hypotheses.extend(str(x)[:200] for x in items[:3])

    return SandboxTask(
        trace_id=trace_id,
        user_query=user_query,
        domains=domains,
        hypotheses=hypotheses,
    )
