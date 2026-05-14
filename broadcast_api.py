import json
import logging
import time
from uuid import uuid4
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

import requests as _requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from run_workflow import run_mycelium_workflow
from api_models import (
    MyceliumRunSummary,
    Layer0Summary,
    RoutingSummary,
    ExpertDecisionSummary,
    Phase2Summary,
    Phase3Summary,
    metrics_to_summary,
    ReasoningTrace,
    append_trace,
    _to_jsonable,
    TRACES_DIR,
)
from conversation_agent import get_conversation_agent
from patch_batch_logger import patch_logger
from sandbox_manager import get_sandbox_manager
from sandbox_models import build_sandbox_task_from_run
import config_loader as cfg

logger = logging.getLogger(__name__)

app = FastAPI(title="Mycelium Broadcast API", version="0.5.0")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    text: str


class SandboxStepOut(BaseModel):
    tool: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    commentary: str
    status: str
    duration_ms: float


class SandboxResultOut(BaseModel):
    trace_id: str
    summary: str
    steps: List[SandboxStepOut]
    started_at: str
    finished_at: str


class ChatResponse(BaseModel):
    trace: MyceliumRunSummary
    answer: str
    sandbox: SandboxResultOut


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse_event(phase: str, detail: str = "", elapsed_ms: int = 0, payload: Any = None) -> str:
    """Serialise one SSE data line."""
    obj: Dict[str, Any] = {"phase": phase, "detail": detail, "elapsed_ms": elapsed_ms}
    if payload is not None:
        obj["payload"] = payload
    return f"data: {json.dumps(obj)}\n\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_run_summary(
    req_text: str,
    trace_id: str,
) -> MyceliumRunSummary:
    """Run the Mycelium workflow and assemble a typed MyceliumRunSummary."""
    all_data, metrics = run_mycelium_workflow([req_text], trace_id=trace_id)
    record = all_data[0]

    layer0_raw: Dict[str, Any] = record.get("layer0_routing", {}) or {}
    layer0 = Layer0Summary(route=layer0_raw.get("route"), raw=layer0_raw)

    routing_raw: Dict[str, Any] = record.get("routing_context", {}) or {}
    routing = RoutingSummary(
        classification=routing_raw.get("classification"),
        selected_domains=list(routing_raw.get("selected_domains", []) or []),
        raw=routing_raw,
    )

    expert_raw: Dict[str, Any] = _to_jsonable(record.get("expert_decision", {}) or {})
    expert_decision = ExpertDecisionSummary(
        decision_type=expert_raw.get("decision_type"),
        selected_experts=list(expert_raw.get("selected_experts", []) or []),
        expert_confidence=expert_raw.get("expert_confidence"),
        raw=expert_raw,
    )

    phase2_raw = _to_jsonable(record.get("phase2_result", {}) or {})
    phase3_raw = _to_jsonable(record.get("phase3_result", {}) or {})
    validation_decision = (
        phase3_raw.get("validation_result")
        or phase3_raw.get("validation_decision")
        or {}
    )
    action_result = phase3_raw.get("action_result", {}) or {}
    phase_latencies = phase3_raw.get("phase_latencies", {}) or {}

    phase2 = Phase2Summary(raw=phase2_raw)
    phase3 = Phase3Summary(
        validation_decision=validation_decision,
        action_result=action_result,
        phase_latencies_ms=phase_latencies,
        raw=phase3_raw,
    )

    metrics_summary = metrics_to_summary(metrics)
    now = datetime.utcnow()

    return MyceliumRunSummary(
        trace_id=trace_id,
        timestamp=now,
        sentence=record.get("sentence", req_text),
        layer0=layer0,
        routing=routing,
        expert_decision=expert_decision,
        phase2=phase2,
        phase3=phase3,
        metrics=metrics_summary,
    )


def _run_sandbox(
    summary: MyceliumRunSummary,
    user_query: str,
    on_progress: Optional[Any] = None,
) -> Any:
    """Run the sandbox for a completed pipeline run.  Never raises."""
    try:
        manager = get_sandbox_manager()
        task = build_sandbox_task_from_run(summary, user_query=user_query)
        return manager.run(task, on_progress=on_progress)
    except Exception as exc:
        logger.warning("Sandbox run failed — %s", exc)
        from sandbox_models import SandboxResult
        now = datetime.utcnow()
        return SandboxResult(
            trace_id=summary.trace_id,
            started_at=now,
            finished_at=now,
            steps=[],
            summary=f"Sandbox unavailable: {exc}",
        )


def _sandbox_result_to_out(sr: Any) -> SandboxResultOut:
    """Convert a SandboxResult to the API output model."""
    steps_out: List[SandboxStepOut] = []
    for s in (sr.steps or []):
        started = s.started_at
        finished = s.finished_at
        duration_ms = (
            (finished - started).total_seconds() * 1000
            if finished and started else 0.0
        )
        steps_out.append(SandboxStepOut(
            tool=s.tool,
            input=s.input,
            output=s.output,
            commentary=s.commentary,
            status=s.status,
            duration_ms=round(duration_ms, 1),
        ))
    return SandboxResultOut(
        trace_id=sr.trace_id,
        summary=sr.summary,
        steps=steps_out,
        started_at=sr.started_at.isoformat() if sr.started_at else "",
        finished_at=sr.finished_at.isoformat() if sr.finished_at else "",
    )


def _full_pipeline_generator(
    req_text: str,
    trace_id: str,
) -> Generator[str, None, None]:
    """
    Run the full chat pipeline and yield SSE events at each phase boundary.

    Phase sequence
    --------------
    routing          — Mycelium multi-lens router + Phase 1/2 running
    expert_decision  — unified decision analysis + expert selection done
    sandbox_plan     — DomainToolPlanner produced tool plan
    sandbox_tool/<n> — each MCP tool call completed
    sandbox_summary  — LLM evidence summary complete
    conversation     — ConversationAgent generating user-facing answer
    done             — complete ChatResponse payload attached
    """
    wall_start = time.monotonic()

    def elapsed() -> int:
        return int((time.monotonic() - wall_start) * 1000)

    # ── Phase: routing + pipeline ───────────────────────────────────────────
    yield _sse_event("routing", "Running multi-lens router and reasoning pipeline…", elapsed())
    logger.info("[SSE %s] phase=routing", trace_id)

    try:
        summary = _build_run_summary(req_text, trace_id)
    except Exception as exc:
        logger.error("[SSE %s] pipeline failed — %s", trace_id, exc)
        yield _sse_event("error", f"Pipeline error: {exc}", elapsed())
        return

    decision_type = (
        summary.expert_decision.decision_type
        if summary.expert_decision else None
    )
    domains = list(summary.routing.selected_domains) if summary.routing else []
    logger.info(
        "[SSE %s] phase=expert_decision type=%s domains=%s elapsed=%dms",
        trace_id, decision_type, domains, elapsed(),
    )
    yield _sse_event(
        "expert_decision",
        f"{decision_type or 'unknown'} · domains: {', '.join(domains) or 'none'}",
        elapsed(),
    )

    is_patch_query = decision_type == "CREATE_NEW_PATCH"
    if is_patch_query:
        try:
            patch_logger.log_query(
                trace_id=trace_id,
                query=req_text,
                tags=domains,
                routing_classification=summary.routing.classification or "",
                phase_latencies_ms=dict(summary.phase3.phase_latencies_ms),
                metadata={
                    "expert_decision_type": decision_type,
                    "expert_confidence": summary.expert_decision.expert_confidence,
                },
            )
        except Exception as exc:
            logger.warning("patch_logger.log_query failed — %s", exc)

    # ── Phase: sandbox ──────────────────────────────────────────────────────
    sandbox_events: List[str] = []
    tool_counter = {"n": 0}

    def on_sandbox_progress(phase: str, detail: str) -> None:
        """Callback fired by SandboxManager at each sub-step."""
        nonlocal sandbox_events
        sandbox_events.append(_sse_event(phase, detail, elapsed()))
        logger.info("[SSE %s] phase=%s detail=%r elapsed=%dms", trace_id, phase, detail, elapsed())

    sandbox_result = _run_sandbox(summary, req_text, on_progress=on_sandbox_progress)

    # Drain any events the sandbox queued synchronously
    for ev in sandbox_events:
        yield ev
    sandbox_events.clear()

    yield _sse_event(
        "sandbox_summary",
        f"Sandbox complete — {len(sandbox_result.steps)} tool call(s)",
        elapsed(),
    )
    logger.info(
        "[SSE %s] phase=sandbox_summary steps=%d elapsed=%dms",
        trace_id, len(sandbox_result.steps), elapsed(),
    )

    # ── Phase: conversation ─────────────────────────────────────────────────
    yield _sse_event("conversation", "Generating answer…", elapsed())
    logger.info("[SSE %s] phase=conversation elapsed=%dms", trace_id, elapsed())

    agent = get_conversation_agent()
    try:
        answer = agent.answer(req_text, summary, sandbox_result=sandbox_result)
    except Exception as exc:
        logger.warning("ConversationAgent failed — %s", exc)
        answer = "Mycelium processed your query but the conversational agent is currently unavailable."

    logger.info("[SSE %s] phase=conversation done elapsed=%dms", trace_id, elapsed())

    # ── Persist trace ───────────────────────────────────────────────────────
    sandbox_dict = _to_jsonable(sandbox_result.model_dump())
    trace = ReasoningTrace(
        trace_id=trace_id,
        timestamp=summary.timestamp,
        user_query=req_text,
        run_summary=summary,
        sandbox_result=sandbox_dict,
    )
    append_trace(trace)

    if is_patch_query and answer:
        try:
            patch_logger.fill_response(trace_id=trace_id, response=answer)
        except Exception as exc:
            logger.warning("patch_logger.fill_response failed — %s", exc)

    # ── Phase: done — send complete payload ─────────────────────────────────
    sandbox_out = _sandbox_result_to_out(sandbox_result)
    response_payload = ChatResponse(
        trace=summary, answer=answer, sandbox=sandbox_out
    ).model_dump()

    logger.info("[SSE %s] phase=done total_elapsed=%dms", trace_id, elapsed())
    yield _sse_event("done", "", elapsed(), payload=_to_jsonable(response_payload))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check — also probes Ollama reachability."""
    ollama_url = cfg.ollama_base_url()
    ollama_ok = False
    try:
        r = _requests.get(ollama_url.rstrip("/") + "/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok",
        "ollama_reachable": ollama_ok,
        "ollama_url": ollama_url,
    }


@app.post("/api/v1/query", response_model=MyceliumRunSummary)
async def query(req: QueryRequest) -> MyceliumRunSummary:
    """Run a single-sentence Mycelium workflow and return a typed summary."""
    trace_id = str(uuid4())
    summary = _build_run_summary(req.text, trace_id)

    sandbox_result = _run_sandbox(summary, req.text)
    sandbox_dict = _to_jsonable(sandbox_result.model_dump())

    trace = ReasoningTrace(
        trace_id=trace_id,
        timestamp=summary.timestamp,
        user_query=req.text,
        run_summary=summary,
        sandbox_result=sandbox_dict,
    )
    append_trace(trace)
    return summary


@app.post("/api/query")
async def query_v0(req: QueryRequest) -> Dict[str, Any]:
    """Backwards-compatible endpoint."""
    summary = await query(req)
    return summary.model_dump()


@app.get("/api/v1/traces/recent", response_model=List[ReasoningTrace])
async def recent_traces(limit: int = 20) -> List[ReasoningTrace]:
    """Return up to `limit` most recent traces."""
    traces: List[ReasoningTrace] = []
    files = sorted(TRACES_DIR.glob("*.jsonl"), reverse=True)
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trace = ReasoningTrace.model_validate_json(line)
                traces.append(trace)
                if len(traces) >= limit:
                    return traces
    return traces


@app.get("/api/v1/traces/{trace_id}", response_model=ReasoningTrace)
async def get_trace(trace_id: str) -> ReasoningTrace:
    """Look up a single trace by ID."""
    files = sorted(TRACES_DIR.glob("*.jsonl"), reverse=True)
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("trace_id") == trace_id:
                    return ReasoningTrace.model_validate(data)
    raise HTTPException(status_code=404, detail="Trace not found")


@app.get("/api/v1/chat/stream")
async def chat_stream(text: str) -> StreamingResponse:
    """
    SSE endpoint — streams one JSON event per pipeline phase.

    Event shape: { phase, detail, elapsed_ms, payload? }

    Phases emitted (in order):
      routing          — pipeline starting
      expert_decision  — decision_type + domains resolved
      sandbox_plan     — tool plan produced
      sandbox_tool/<n> — each MCP call completed
      sandbox_summary  — evidence summary complete
      conversation     — generating answer
      done             — complete ChatResponse in `payload`
      error            — fatal error in `detail`

    The frontend connects via EventSource, updates the phase indicator,
    then reads the final `done` payload as the chat response.
    """
    if not text or not text.strip():
        async def _empty():
            yield _sse_event("error", "Empty query", 0)
        return StreamingResponse(_empty(), media_type="text/event-stream")

    trace_id = str(uuid4())
    logger.info("[SSE] new stream trace_id=%s query=%r", trace_id, text[:80])

    def _sync_gen():
        yield from _full_pipeline_generator(text.strip(), trace_id)

    # Run the synchronous generator inside a thread so the event loop isn't blocked
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _sentinel = object()

    def _producer():
        try:
            for chunk in _sync_gen():
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _sentinel)

    executor.submit(_producer)

    async def _async_gen():
        while True:
            item = await queue.get()
            if item is _sentinel:
                break
            yield item

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Full pipeline: Mycelium reasoning + sandbox research + conversational explanation.

    Identical to the SSE stream but returned as a single JSON response.
    Useful as fallback when EventSource is unavailable.

    Flow:
      1. Run the Mycelium reasoning pipeline.
      2. If CREATE_NEW_PATCH: log the query to the patch-batch dataset (M1.5).
      3. Run the sandbox (M3): gather evidence via tools.
      4. Pass sandbox evidence to the conversational agent (M4).
      5. Persist the full ReasoningTrace (with sandbox_result) to JSONL (M2).
      6. If CREATE_NEW_PATCH: fill the patch-batch record with the final answer.
    """
    trace_id = str(uuid4())
    summary = _build_run_summary(req.text, trace_id)

    decision_type = (
        summary.expert_decision.decision_type
        if summary.expert_decision else None
    )
    is_patch_query = decision_type == "CREATE_NEW_PATCH"

    if is_patch_query:
        try:
            patch_logger.log_query(
                trace_id=trace_id,
                query=req.text,
                tags=list(summary.routing.selected_domains),
                routing_classification=summary.routing.classification or "",
                phase_latencies_ms=dict(summary.phase3.phase_latencies_ms),
                metadata={
                    "expert_decision_type": decision_type,
                    "expert_confidence": summary.expert_decision.expert_confidence,
                },
            )
        except Exception as exc:
            logger.warning("patch_logger.log_query failed — %s", exc)

    sandbox_result = _run_sandbox(summary, req.text)
    sandbox_dict = _to_jsonable(sandbox_result.model_dump())

    agent = get_conversation_agent()
    try:
        answer = agent.answer(req.text, summary, sandbox_result=sandbox_result)
    except Exception as exc:
        logger.warning("ConversationAgent failed — %s", exc)
        answer = "Mycelium processed your query but the conversational agent is currently unavailable."

    trace = ReasoningTrace(
        trace_id=trace_id,
        timestamp=summary.timestamp,
        user_query=req.text,
        run_summary=summary,
        sandbox_result=sandbox_dict,
    )
    append_trace(trace)

    if is_patch_query and answer:
        try:
            patch_logger.fill_response(trace_id=trace_id, response=answer)
        except Exception as exc:
            logger.warning("patch_logger.fill_response failed — %s", exc)

    sandbox_out = _sandbox_result_to_out(sandbox_result)
    return ChatResponse(trace=summary, answer=answer, sandbox=sandbox_out)
