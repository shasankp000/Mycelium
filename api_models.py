"""
api_models.py  —  Pydantic schemas for Mycelium's HTTP + SSE API.

Milestone 1: Formalize MyceliumRunSummary and ChatResponse so that
  - post_check fields (verified_answer, verification_meta, post_check_degraded)
    are first-class response members, not buried in opaque metadata dicts.
  - The SSE /api/v1/chat/stream endpoint can type-safely emit a 'done' payload.
  - /api/v1/traces/recent can deserialise archived JSONL records.

All models use model_config = ConfigDict(extra='allow') so that callers
passing additional fields (e.g. phase_latencies_ms sub-keys) do not raise
ValidationError — we only assert on fields we actually read.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


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


class Phase3Summary(BaseModel):
    model_config = ConfigDict(extra="allow")
    validation_decision: Optional[ValidationDecision] = None
    phase_latencies_ms: Optional[Dict[str, float]] = Field(default_factory=dict)
    answer_draft: Optional[str] = None


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
    """
    model_config = ConfigDict(extra="allow")

    trace_id: Optional[str] = None
    timestamp: Optional[str] = None
    sentence: Optional[str] = None                     # original user query

    # Pipeline stages
    layer0: Optional[Layer0Summary] = None
    routing: Optional[RoutingSummary] = None
    expert_decision: Optional[ExpertDecisionSummary] = None
    phase3: Optional[Phase3Summary] = None

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

class HistoryTrace(BaseModel):
    """
    One record stored in traces/<trace_id>.jsonl and returned by
    GET /api/v1/traces/recent.
    """
    model_config = ConfigDict(extra="allow")

    trace_id: str
    timestamp: str
    user_query: str
    run_summary: MyceliumRunSummary
    sandbox_result: Optional[Dict[str, Any]] = None


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
