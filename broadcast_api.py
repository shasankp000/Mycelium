import json
from uuid import uuid4
from datetime import datetime
from typing import Any, Dict, List

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


app = FastAPI(title="Mycelium Broadcast API", version="0.3.0")

# Allow local Next.js dev server by default; can be restricted in prod
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


class ChatResponse(BaseModel):
    trace: MyceliumRunSummary
    answer: str


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/v1/query", response_model=MyceliumRunSummary)
async def query(req: QueryRequest) -> MyceliumRunSummary:
    """Run a single-sentence Mycelium workflow and return a typed summary."""

    trace_id = str(uuid4())
    all_data, metrics = run_mycelium_workflow([req.text], trace_id=trace_id)
    record = all_data[0]

    # Layer0
    layer0_raw: Dict[str, Any] = record.get("layer0_routing", {}) or {}
    layer0 = Layer0Summary(
        route=layer0_raw.get("route"),
        raw=layer0_raw,
    )

    # Routing
    routing_raw: Dict[str, Any] = record.get("routing_context", {}) or {}
    routing = RoutingSummary(
        classification=routing_raw.get("classification"),
        selected_domains=list(routing_raw.get("selected_domains", []) or []),
        raw=routing_raw,
    )

    # Expert decision
    expert_raw: Dict[str, Any] = _to_jsonable(
        record.get("expert_decision", {}) or {}
    )
    decision_type = expert_raw.get("decision_type")
    selected_experts = list(expert_raw.get("selected_experts", []) or [])
    expert_confidence = expert_raw.get("expert_confidence")
    expert_decision = ExpertDecisionSummary(
        decision_type=decision_type,
        selected_experts=selected_experts,
        expert_confidence=expert_confidence,
        raw=expert_raw,
    )

    # Phase 2 / 3
    phase2_raw = _to_jsonable(record.get("phase2_result", {}) or {})
    phase3_raw = _to_jsonable(record.get("phase3_result", {}) or {})

    # FIX (Bug C): SystemExecutionResult stores the validation decision
    # under "validation_result", not "validation_decision".  The wrong key
    # caused Phase3Summary.validation_decision to always be {}, which is why
    # the UI showed "no specific validation result recorded".
    # We try both keys for forwards/backwards compatibility.
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

    summary = MyceliumRunSummary(
        trace_id=trace_id,
        timestamp=now,
        sentence=record.get("sentence", req.text),
        layer0=layer0,
        routing=routing,
        expert_decision=expert_decision,
        phase2=phase2,
        phase3=phase3,
        metrics=metrics_summary,
    )

    # Persist a basic reasoning trace for this run
    trace = ReasoningTrace(
        trace_id=trace_id,
        timestamp=now,
        user_query=req.text,
        run_summary=summary,
    )
    append_trace(trace)

    return summary


# Backwards-compatible endpoint for the existing UI while we migrate
@app.post("/api/query")
async def query_v0(req: QueryRequest) -> Dict[str, Any]:
    summary = await query(req)
    return summary.model_dump()


@app.get("/api/v1/traces/recent", response_model=List[ReasoningTrace])
async def recent_traces(limit: int = 20) -> List[ReasoningTrace]:
    """Return up to `limit` most recent traces (naive JSONL scan)."""

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
    """Look up a single trace by ID via JSONL scan.

    This is intentionally naive but fine for PoC volumes.
    """

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
    """Run Mycelium for the input and have the conversational agent explain it.

    If the pipeline decision was CREATE_NEW_PATCH, the conversational agent's
    answer is also patched back into the batch log so the record is complete
    for offline training.
    """

    summary = await query(QueryRequest(text=req.text))
    agent = get_conversation_agent()
    answer = agent.answer(req.text, summary)

    # If this was a no-domain (patch) query, fill the response into the log.
    # The log_query() call already happened inside run_mycelium_workflow;
    # fill_response() is idempotent if the pipeline already filled it.
    if (
        summary.expert_decision
        and summary.expert_decision.decision_type == "CREATE_NEW_PATCH"
        and answer
    ):
        patch_logger.fill_response(
            trace_id=summary.trace_id,
            response=answer,
        )

    return ChatResponse(trace=summary, answer=answer)
