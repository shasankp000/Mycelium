import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from datetime import datetime, UTC
from typing import Any, Dict, Generator, List, Optional

import requests as _requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mycelium.pipeline.run_workflow import run_mycelium_workflow
from mycelium.pipeline.api_models import (
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
from mycelium.pipeline.conversation_agent import get_conversation_agent
from mycelium.pipeline.patch_batch_logger import patch_logger
from mycelium.pipeline.sandbox_manager import get_sandbox_manager
from mycelium.pipeline.sandbox_models import build_sandbox_task_from_run
from mycelium.pipeline.pipeline_event import ReplayJournal
import mycelium.pipeline.config_loader as cfg

logger = logging.getLogger(__name__)

app = FastAPI(title="Mycelium Broadcast API", version="0.6.0")

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

_warmup_done = threading.Event()
_warmup_lock = threading.Lock()


@app.on_event("startup")
async def preload_models() -> None:
    import asyncio
    from model_registry import warmup

    specs = [
        {
            "model_name": "all-mpnet-base-v2",
            "model_type": "sentence_transformer",
            "device": "cpu",
        },
        {
            "model_name": "all-MiniLM-L6-v2",
            "model_type": "sentence_transformer",
            "device": "cpu",
        },
    ]

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        await loop.run_in_executor(pool, warmup, specs)

    _warmup_done.set()
    logger.info("broadcast_api: model warmup complete — _warmup_done set")


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


def _sse_event(
    phase: str, detail: str = "", elapsed_ms: int = 0, payload: Any = None
) -> str:
    obj: Dict[str, Any] = {"phase": phase, "detail": detail, "elapsed_ms": elapsed_ms}
    if payload is not None:
        obj["payload"] = payload
    return f"data: {json.dumps(obj)}\n\n"


def _pipeline_event_to_sse(ev_dict: Dict[str, Any]) -> str:
    seq = ev_dict.get("sequence_number", 0)
    data = json.dumps(ev_dict)
    return f"id: {seq}\ndata: {data}\n\n"


def _build_run_summary(
    req_text: str,
    trace_id: str,
    on_event: Optional[Any] = None,
) -> MyceliumRunSummary:
    all_data, metrics = run_mycelium_workflow(
        [req_text], trace_id=trace_id, on_event=on_event
    )
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
    now = datetime.now(UTC)

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
    try:
        manager = get_sandbox_manager()
        task = build_sandbox_task_from_run(summary, user_query=user_query)
        return manager.run(task, on_progress=on_progress)
    except Exception as exc:
        logger.warning("Sandbox run failed — %s", exc)
        from mycelium.pipeline.sandbox_models import SandboxResult

        now = datetime.now(UTC)
        return SandboxResult(
            trace_id=summary.trace_id,
            started_at=now,
            finished_at=now,
            steps=[],
            summary=f"Sandbox unavailable: {exc}",
        )


def _sandbox_result_to_out(sr: Any) -> SandboxResultOut:
    steps_out: List[SandboxStepOut] = []
    for s in sr.steps or []:
        started = s.started_at
        finished = s.finished_at
        duration_ms = (
            (finished - started).total_seconds() * 1000 if finished and started else 0.0
        )
        steps_out.append(
            SandboxStepOut(
                tool=s.tool,
                input=s.input,
                output=s.output,
                commentary=s.commentary,
                status=s.status,
                duration_ms=round(duration_ms, 1),
            )
        )
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
    sse_queue: Optional[Any] = None,
    loop: Optional[Any] = None,
    replay_journal: Optional[ReplayJournal] = None,
) -> Generator[str, None, None]:
    import concurrent.futures as _cf

    wall_start = time.monotonic()

    def elapsed() -> int:
        return int((time.monotonic() - wall_start) * 1000)

    sent_setting_up = False
    if not _warmup_done.is_set():
        sent_setting_up = True
        yield _sse_event(
            "setting_up",
            "Loading model weights into memory — first request may take a moment\u2026",
            elapsed(),
        )
        logger.info("[SSE %s] phase=setting_up — waiting for warmup", trace_id)
        _warmup_done.wait()

    if sent_setting_up:
        yield _sse_event(
            "environment_ready", "Environment ready — starting pipeline", elapsed()
        )
        logger.info(
            "[SSE %s] phase=environment_ready elapsed=%dms", trace_id, elapsed()
        )

    def on_pipeline_event(ev_dict: Dict[str, Any]) -> None:
        sse_line = _pipeline_event_to_sse(ev_dict)
        if replay_journal is not None:
            from mycelium.pipeline.pipeline_event import PipelineEvent as _PE

            _shell = _PE(
                event_id=ev_dict.get("event_id", ""),
                request_id=ev_dict.get("request_id", ""),
                sequence_number=ev_dict.get("sequence_number", 0),
                timestamp=ev_dict.get("timestamp", 0.0),
                phase_id=ev_dict.get("phase_id", 0),
                phase_name=ev_dict.get("phase_name", ""),
                substep=ev_dict.get("substep", ""),
                state=ev_dict.get("state", "running"),
                visibility=ev_dict.get("visibility", "public"),
                message=ev_dict.get("message", ""),
                detail=ev_dict.get("detail", ""),
                elapsed_ms=ev_dict.get("elapsed_ms", 0.0),
                metadata=ev_dict.get("metadata", {}),
            )
            replay_journal.record(_shell)
        logger.debug(
            "[SSE %s] pipeline_event phase=%s seq=%d",
            trace_id,
            ev_dict.get("phase_name"),
            ev_dict.get("sequence_number"),
        )
        if sse_queue is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(sse_queue.put_nowait, sse_line)
            except RuntimeError:
                pass
        else:
            _pipeline_event_buf.append(sse_line)

    _pipeline_event_buf: List[str] = []

    logger.info("[SSE %s] phase=routing — submitting workflow to thread", trace_id)

    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as _exec:
            _future = _exec.submit(
                _build_run_summary, req_text, trace_id, on_pipeline_event
            )
            while not _future.done():
                time.sleep(15)
                if _future.done():
                    break
                yield ": heartbeat\n\n"
                logger.debug("[SSE %s] heartbeat elapsed=%dms", trace_id, elapsed())
            summary = _future.result()
    except Exception as exc:
        logger.error("[SSE %s] pipeline failed — %s", trace_id, exc)
        yield _sse_event("error", f"Pipeline error: {exc}", elapsed())
        return

    for ev_line in _pipeline_event_buf:
        yield ev_line
    _pipeline_event_buf.clear()

    decision_type = (
        summary.expert_decision.decision_type if summary.expert_decision else None
    )
    domains = list(summary.routing.selected_domains) if summary.routing else []
    logger.info(
        "[SSE %s] phase=expert_decision type=%s domains=%s elapsed=%dms",
        trace_id,
        decision_type,
        domains,
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

    sandbox_buffer: List[str] = []

    def on_sandbox_progress(phase: str, detail: str) -> None:
        ev = _sse_event(phase, detail, elapsed())
        logger.info(
            "[SSE %s] phase=%s detail=%r elapsed=%dms",
            trace_id,
            phase,
            detail,
            elapsed(),
        )
        if sse_queue is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(sse_queue.put_nowait, ev)
            except RuntimeError:
                sandbox_buffer.append(ev)
        else:
            sandbox_buffer.append(ev)

    sandbox_result = _run_sandbox(summary, req_text, on_progress=on_sandbox_progress)

    for ev in sandbox_buffer:
        yield ev
    sandbox_buffer.clear()

    yield _sse_event(
        "sandbox_summary",
        f"Sandbox complete — {len(sandbox_result.steps)} tool call(s)",
        elapsed(),
    )
    logger.info(
        "[SSE %s] phase=sandbox_summary steps=%d elapsed=%dms",
        trace_id,
        len(sandbox_result.steps),
        elapsed(),
    )

    yield _sse_event("conversation", "Generating answer\u2026", elapsed())
    logger.info("[SSE %s] phase=conversation elapsed=%dms", trace_id, elapsed())

    agent = get_conversation_agent()
    try:
        answer = agent.answer(req_text, summary, sandbox_result=sandbox_result)
    except Exception as exc:
        logger.warning("ConversationAgent failed — %s", exc)
        answer = "I processed your query but my conversational agent is currently unavailable."

    logger.info("[SSE %s] phase=conversation done elapsed=%dms", trace_id, elapsed())

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
            sandbox_evidence = sandbox_dict.get("steps")
            patch_logger.fill_response(
                trace_id=trace_id,
                response=answer,
                sandbox_evidence=sandbox_evidence
                if isinstance(sandbox_evidence, list)
                else None,
                phase_latencies_ms=dict(summary.phase3.phase_latencies_ms),
            )
        except Exception as exc:
            logger.warning("patch_logger.fill_response failed — %s", exc)

    sandbox_out = _sandbox_result_to_out(sandbox_result)
    response_payload = ChatResponse(
        trace=summary, answer=answer, sandbox=sandbox_out
    ).model_dump()
    logger.info("[SSE %s] phase=done total_elapsed=%dms", trace_id, elapsed())
    yield _sse_event("done", "", elapsed(), payload=_to_jsonable(response_payload))


@app.get("/health")
async def health() -> Dict[str, Any]:
    from model_registry import loaded_models

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
        "models_loaded": loaded_models(),
        "warmup_done": _warmup_done.is_set(),
    }


@app.get("/api/v1/health")
async def health_v1() -> Dict[str, Any]:
    return await health()


@app.post("/api/v1/query", response_model=MyceliumRunSummary)
async def query(req: QueryRequest) -> MyceliumRunSummary:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    trace_id = str(uuid4())
    with ThreadPoolExecutor(max_workers=1) as pool:
        summary = await loop.run_in_executor(
            pool, _build_run_summary, req.text, trace_id, None
        )
    sandbox_result = await loop.run_in_executor(None, _run_sandbox, summary, req.text)
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
    summary = await query(req)
    return summary.model_dump()


@app.get("/api/v1/traces/recent", response_model=List[ReasoningTrace])
async def recent_traces(limit: int = 20) -> List[ReasoningTrace]:
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


_replay_journals: Dict[str, ReplayJournal] = {}
_replay_journals_lock = threading.Lock()


@app.get("/api/v1/chat/stream")
async def chat_stream(text: str, request: Request) -> StreamingResponse:
    if not text or not text.strip():

        async def _empty():
            yield _sse_event("error", "Empty query", 0)

        return StreamingResponse(_empty(), media_type="text/event-stream")

    trace_id = str(uuid4())
    logger.info("[SSE] new stream trace_id=%s query=%r", trace_id, text[:80])

    journal = ReplayJournal(maxlen=200)
    with _replay_journals_lock:
        _replay_journals[trace_id] = journal

    last_event_id_header = request.headers.get("Last-Event-ID", "")
    last_seen_seq: int = 0
    if last_event_id_header:
        try:
            last_seen_seq = int(last_event_id_header)
        except (ValueError, TypeError):
            last_seen_seq = 0

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _sentinel = object()
    _cancel = threading.Event()

    def _producer(cancel: threading.Event) -> None:
        try:
            for chunk in _full_pipeline_generator(
                text.strip(),
                trace_id,
                sse_queue=queue,
                loop=loop,
                replay_journal=journal,
            ):
                if cancel.is_set():
                    logger.info(
                        "[SSE %s] producer SSE cancelled — pipeline continues", trace_id
                    )
                    continue
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except RuntimeError:
                    logger.warning(
                        "[SSE %s] call_soon_threadsafe: loop closed", trace_id
                    )
                    return
        finally:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, _sentinel)
            except RuntimeError:
                pass
            with _replay_journals_lock:
                _replay_journals.pop(trace_id, None)

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(_producer, _cancel)
    executor.shutdown(wait=False)

    async def _async_gen():
        if last_seen_seq > 0:
            replayed = journal.replay_from(last_seen_seq)
            logger.info(
                "[SSE %s] reconnect: replaying %d missed event(s) after seq=%d",
                trace_id,
                len(replayed),
                last_seen_seq,
            )
            for ev in replayed:
                yield _pipeline_event_to_sse(ev.to_sse_dict())
        try:
            while True:
                item = await queue.get()
                if item is _sentinel:
                    break
                yield item
        except GeneratorExit:
            _cancel.set()
            logger.info(
                "[SSE %s] client disconnected — SSE cancelled, pipeline continues",
                trace_id,
            )

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _warmup_done.wait)
    trace_id = str(uuid4())

    with ThreadPoolExecutor(max_workers=1) as pool:
        summary = await loop.run_in_executor(
            pool, _build_run_summary, req.text, trace_id, None
        )

    decision_type = (
        summary.expert_decision.decision_type if summary.expert_decision else None
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
        answer = "I processed your query but my conversational agent is currently unavailable."

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
