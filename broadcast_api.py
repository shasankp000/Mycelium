import json
import logging
from uuid import uuid4
from datetime import datetime
from typing import Any, Dict, List

import requests as _requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title="Mycelium Broadcast API", version="0.4.0")

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


def _run_sandbox(summary: MyceliumRunSummary, user_query: str) -> Any:
    """Run the sandbox for a completed pipeline run.  Never raises."""
    try:
        manager = get_sandbox_manager()
        task = build_sandbox_task_from_run(summary, user_query=user_query)
        return manager.run(task)
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


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Full pipeline: Mycelium reasoning + sandbox research + conversational explanation.

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

    # M1.5: log the query immediately so the record exists even if later steps fail
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

    # M3: run sandbox
    sandbox_result = _run_sandbox(summary, req.text)
    sandbox_dict = _to_jsonable(sandbox_result.model_dump())

    # M4: generate conversational answer with sandbox context
    agent = get_conversation_agent()
    try:
        answer = agent.answer(req.text, summary, sandbox_result=sandbox_result)
    except Exception as exc:
        logger.warning("ConversationAgent failed — %s", exc)
        answer = "Mycelium processed your query but the conversational agent is currently unavailable."

    # M2: persist full trace (pipeline + sandbox)
    trace = ReasoningTrace(
        trace_id=trace_id,
        timestamp=summary.timestamp,
        user_query=req.text,
        run_summary=summary,
        sandbox_result=sandbox_dict,
    )
    append_trace(trace)

    # M1.5: fill the patch-batch record with the final answer
    if is_patch_query and answer:
        try:
            patch_logger.fill_response(trace_id=trace_id, response=answer)
        except Exception as exc:
            logger.warning("patch_logger.fill_response failed — %s", exc)

    sandbox_out = _sandbox_result_to_out(sandbox_result)
    return ChatResponse(trace=summary, answer=answer, sandbox=sandbox_out)
