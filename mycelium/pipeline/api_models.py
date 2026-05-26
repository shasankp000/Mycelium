"""
api_models.py  —  Pydantic schemas for Mycelium's HTTP + SSE API.

Milestone 1: Formalize MyceliumRunSummary and ChatResponse so that
  - post_check fields (verified_answer, verification_meta, post_check_degraded)
    are first-class response members, not buried in opaque metadata dicts.
  - The SSE /api/v1/chat/stream endpoint can type-safely emit a 'done' payload.
  - /api/v1/traces/recent can deserialise archived JSONL records.

Phase 6: Add ReasoningMode enum and wire it through ChatRequest →
  MyceliumRunSummary so that the graph UI can label each trace with the
  depth that produced it.

Wiring update: modes renamed to fast / smart / researcher to match the
  frontend ModeSelector.  trm_threshold added to every depth config so
  run_workflow can apply per-mode TRM confidence thresholds without
  hardcoding values outside this file.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Reasoning mode  (Phase 6 — wiring update)
# ---------------------------------------------------------------------------

ReasoningMode = Literal["fast", "smart", "researcher"]

# ── Per-mode depth + TRM threshold contract ─────────────────────────────────
#
#   dfs_max_depth   — ceiling on how many DFS layers the pipeline may climb.
#   expert_top_k    — how many experts Phase-2 consults per query.
#   phase3_passes   — number of Phase-3 validation passes.
#   trm_threshold   — TRM halt_confidence must EXCEED this value for the
#                     pipeline to short-circuit without escalating.
#                     Lowering the threshold (researcher) makes TRM escalate
#                     even on moderately confident queries so the full DAG runs.
#
# These values are the contract between TRM checkpoint training targets and
# the runtime.  Adjust here only — run_workflow reads them via get_depth_config.
# ---------------------------------------------------------------------------

DEPTH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "dfs_max_depth": 1,
        "expert_top_k": 1,
        "phase3_passes": 1,
        "trm_threshold": 0.85,   # strict — escalate only when clearly uncertain
    },
    "smart": {
        "dfs_max_depth": 3,
        "expert_top_k": 2,
        "phase3_passes": 2,
        "trm_threshold": 0.70,   # default — normal TRM behaviour
    },
    "researcher": {
        "dfs_max_depth": 5,
        "expert_top_k": 3,
        "phase3_passes": 3,
        "trm_threshold": 0.55,   # loose — escalate eagerly, run full DAG
    },
}

# Backward-compat alias: old "balanced" / "deep" strings map to smart / researcher.
_MODE_ALIASES: Dict[str, str] = {
    "balanced": "smart",
    "deep": "researcher",
}


def get_depth_config(mode: ReasoningMode) -> Dict[str, Any]:
    """Return the depth-config dict for *mode*, defaulting to 'smart'.

    Also accepts legacy aliases ('balanced' → 'smart', 'deep' → 'researcher')
    so that any existing traces or callers don't break.
    """
    resolved = _MODE_ALIASES.get(mode, mode)  # type: ignore[arg-type]
    return DEPTH_CONFIGS.get(resolved, DEPTH_CONFIGS["smart"])


# ---------------------------------------------------------------------------
# Trace storage
# ---------------------------------------------------------------------------

TRACES_DIR = pathlib.Path("traces")
TRACES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class Layer0Summary(BaseModel):
    model_config = ConfigDict(extra="allow")
    route: Optional[str] = None


class RoutingSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    classification: Optional[str] = None
    selected_domains: Optional[List[str]] = Field(default_factory=list)
    routing_confidence: Optional[float] = None


class ExpertDecisionSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    decision_type: Optional[str] = None
    selected_experts: Optional[List[str]] = Field(default_factory=list)
    expert_confidence: Optional[float] = None
    # CREATE_NEW_PATCH specific
    patch_domain_tag: Optional[str] = None
    patch_log_path: Optional[str] = None


class ValidationDecision(BaseModel):
    model_config = ConfigDict(extra="allow")
    result_class: Optional[str] = None
    confidence: Optional[float] = None


class Phase2Summary(BaseModel):
    """
    Summary of Phase-2 fusion / validation results.
    Kept intentionally open (extra='allow') because the fusion engine
    attaches domain-specific sub-keys that vary per query.
    """
    model_config = ConfigDict(extra="allow")
    answer_draft: Optional[str] = None
    fusion_confidence: Optional[float] = None


class Phase3Summary(BaseModel):
    model_config = ConfigDict(extra="allow")
    validation_decision: Optional[ValidationDecision] = None
    phase_latencies_ms: Optional[Dict[str, float]] = Field(default_factory=dict)
    answer_draft: Optional[str] = None


class MetricsSummary(BaseModel):
    """
    Aggregate performance metrics for a single pipeline run.
    Written by run_workflow and surfaced in the trace panel.
    """
    model_config = ConfigDict(extra="allow")
    total_latency_ms: Optional[float] = None
    routing_latency_ms: Optional[float] = None
    expert_latency_ms: Optional[float] = None
    phase3_latency_ms: Optional[float] = None
    post_check_latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Post-check  (Milestone 1 — fields promoted to top-level)
# ---------------------------------------------------------------------------

class PostCheckMeta(BaseModel):
    """
    Verification metadata attached by expert_post_check.runner.
    Surfaced in the UI as a collapsible 'Post-check' section in TracePanel.
    """
    model_config = ConfigDict(extra="allow")

    # TRM (Qwen3.5) assessment
    trm_verdict: Optional[str] = None           # e.g. "VERIFIED", "UNVERIFIED", "TIMEOUT"
    trm_confidence: Optional[float] = None
    trm_latency_ms: Optional[float] = None

    # Fuzzy cosine-sim verifier
    fuzzy_score: Optional[float] = None         # all-mpnet-base-v2 cosine similarity
    fuzzy_verdict: Optional[str] = None         # "PASS" / "FAIL"

    # P6 adapter fallback
    p6_used: Optional[bool] = None
    p6_answer: Optional[str] = None

    # RL weight info
    rl_domain: Optional[str] = None
    rl_weight: Optional[float] = None

    # Degraded flag — set when Qwen3.5 timed out or VRAM was unavailable
    degraded: Optional[bool] = Field(default=False)
    degraded_reason: Optional[str] = None


class VerificationMeta(BaseModel):
    """
    Top-level verification envelope written to ActionResult.metadata
    by PostCheckRunner._execute_use_existing.
    """
    model_config = ConfigDict(extra="allow")
    verified: Optional[bool] = None
    post_check: Optional[PostCheckMeta] = None
    final_answer_source: Optional[str] = None   # "trm" | "p6" | "original" | "degraded"


# ---------------------------------------------------------------------------
# Run summary  (full trace record)
# ---------------------------------------------------------------------------

class MyceliumRunSummary(BaseModel):
    """
    Authoritative record of one Mycelium reasoning run.
    Written to traces/<trace_id>.jsonl and returned inside ChatResponse.

    Phase 6: reasoning_mode echoed back so the graph UI can label
    each snapshot with the depth that produced it.
    """
    model_config = ConfigDict(extra="allow")

    trace_id: Optional[str] = None
    timestamp: Optional[Union[str, datetime]] = None
    sentence: Optional[str] = None                     # original user query

    # Phase 6 — reasoning mode used for this run
    reasoning_mode: ReasoningMode = "smart"

    # Pipeline stages
    layer0: Optional[Layer0Summary] = None
    routing: Optional[RoutingSummary] = None
    expert_decision: Optional[ExpertDecisionSummary] = None
    phase2: Optional[Phase2Summary] = None
    phase3: Optional[Phase3Summary] = None

    # Aggregate metrics
    metrics: Optional[MetricsSummary] = None

    # Post-check results  (Milestone 1 — promoted from opaque metadata)
    verification_meta: Optional[VerificationMeta] = None
    post_check_degraded: Optional[bool] = Field(default=False)

    # CREATE_NEW_PATCH flag  (Milestone 1.5)
    is_patch_query: Optional[bool] = Field(default=False)
    patch_dataset_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Sandbox models  (unchanged from previous version)
# ---------------------------------------------------------------------------

class SandboxStep(BaseModel):
    model_config = ConfigDict(extra="allow")
    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    commentary: str = ""
    status: str = "ok"
    duration_ms: float = 0.0


class SandboxResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    trace_id: str = ""
    summary: str = ""
    steps: List[SandboxStep] = Field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""


# ---------------------------------------------------------------------------
# Chat request / response
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    # Phase 6 — reasoning depth selected by the user in the UI
    reasoning_mode: ReasoningMode = "smart"


class ChatResponse(BaseModel):
    """
    Response body for POST /api/v1/chat  and  the 'done' SSE payload.

    answer             — final natural-language answer
    verified_answer    — post-checked / TRM-verified answer (may differ from answer
                         when P6 produced a better result; otherwise identical)
    trace              — full run summary for TracePanel
    sandbox            — optional sandbox evidence
    post_check_degraded — True when TRM timed out; UI shows amber warning badge
    """
    model_config = ConfigDict(extra="allow")

    answer: Optional[str] = None
    verified_answer: Optional[str] = None
    trace: Optional[MyceliumRunSummary] = None
    sandbox: Optional[SandboxResult] = None
    post_check_degraded: bool = False


# ---------------------------------------------------------------------------
# SSE event envelope
# ---------------------------------------------------------------------------

class SseEvent(BaseModel):
    """
    Shape of every JSON object emitted on the /api/v1/chat/stream endpoint.

    Phases emitted in order:
      setting_up → environment_ready → routing → expert_decision →
      [sandbox_plan → sandbox_tool/<n> → sandbox_tool/<n>_ok|err …] →
      conversation → done  |  error

    The 'done' phase carries the full ChatResponse as payload.
    """
    phase: str
    detail: str = ""
    elapsed_ms: float = 0.0
    payload: Optional[ChatResponse] = None


# ---------------------------------------------------------------------------
# Trace archive  (used by /api/v1/traces/recent)
# ---------------------------------------------------------------------------

class ReasoningTrace(BaseModel):
    """
    One record stored in traces/<trace_id>.jsonl and returned by
    GET /api/v1/traces/recent.

    Previously called HistoryTrace; renamed to ReasoningTrace to match
    the name used throughout broadcast_api.py.
    """
    model_config = ConfigDict(extra="allow")

    trace_id: str
    timestamp: Union[str, datetime]
    user_query: str
    run_summary: MyceliumRunSummary
    sandbox_result: Optional[Dict[str, Any]] = None


# Keep the old name as an alias so any code still referencing HistoryTrace works.
HistoryTrace = ReasoningTrace


# ---------------------------------------------------------------------------
# Patch dataset record  (Milestone 1.5)
# ---------------------------------------------------------------------------

class PatchRecord(BaseModel):
    """
    Written to patch_dataset/<domain_tag>/<date>.jsonl when
    expert_decision.decision_type == 'CREATE_NEW_PATCH'.

    The record is written *before* the pipeline runs (with final_answer=null)
    and updated *after* with the actual answer for future patch training.
    """
    model_config = ConfigDict(extra="allow")

    trace_id: str
    timestamp: str
    domain_tag: str
    user_query: str
    routing_domains: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None         # filled in after pipeline completes
    sandbox_summary: Optional[str] = None
    confidence_at_routing: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers used by broadcast_api.py
# ---------------------------------------------------------------------------

def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serialisable form.

    Handles: datetime → ISO string, Pydantic models → dict,
    sets → list, and arbitrary nested dicts/lists.
    """
    if obj is None:
        return None
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return _to_jsonable(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(i) for i in obj]
    if isinstance(obj, set):
        return [_to_jsonable(i) for i in sorted(obj, key=str)]
    return obj


def metrics_to_summary(metrics: Any) -> Optional[MetricsSummary]:
    """
    Convert a raw metrics dict (or object) returned by run_workflow into a
    typed MetricsSummary.  Returns None if metrics is falsy.
    """
    if not metrics:
        return None
    if isinstance(metrics, MetricsSummary):
        return metrics
    if isinstance(metrics, BaseModel):
        data = metrics.model_dump()
    elif isinstance(metrics, dict):
        data = metrics
    else:
        return None
    return MetricsSummary(**{k: v for k, v in data.items() if v is not None})


def append_trace(trace: ReasoningTrace) -> None:
    """
    Append a ReasoningTrace record to TRACES_DIR/<trace_id>.jsonl.

    Each file holds exactly one JSONL line so that the file can be quickly
    located and the record streamed without loading the whole archive.
    Uses _to_jsonable so datetime fields are safely serialised.
    """
    path = TRACES_DIR / f"{trace.trace_id}.jsonl"
    record = _to_jsonable(trace.model_dump())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
