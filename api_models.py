from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from run_workflow import WorkflowMetrics


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of nested objects to JSON-serialisable values.

    Mirrors the helper used in run_workflow so the API schema stays stable
    even if internals change slightly.
    """

    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return _to_jsonable(vars(obj))
    return obj


class Layer0Summary(BaseModel):
    route: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class RoutingSummary(BaseModel):
    classification: Optional[str] = None
    selected_domains: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class ExpertDecisionSummary(BaseModel):
    decision_type: Optional[str] = None
    selected_experts: List[str] = Field(default_factory=list)
    expert_confidence: Optional[float] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class Phase2Summary(BaseModel):
    raw: Dict[str, Any] = Field(default_factory=dict)


class Phase3Summary(BaseModel):
    validation_decision: Dict[str, Any] = Field(default_factory=dict)
    action_result: Dict[str, Any] = Field(default_factory=dict)
    phase_latencies_ms: Dict[str, float] = Field(default_factory=dict)
    raw: Dict[str, Any] = Field(default_factory=dict)


class MetricsSummary(BaseModel):
    layer0_routes: Dict[str, int] = Field(default_factory=dict)
    routing_classifications: Dict[str, int] = Field(default_factory=dict)
    expert_decisions: Dict[str, int] = Field(default_factory=dict)
    domains: Dict[str, int] = Field(default_factory=dict)


class MyceliumRunSummary(BaseModel):
    """High-level, stable schema for one Mycelium run.

    This is what the web API exposes and what future layers (sandbox,
    conversational agent) will consume.
    """

    trace_id: str
    timestamp: datetime
    sentence: str
    layer0: Layer0Summary
    routing: RoutingSummary
    expert_decision: ExpertDecisionSummary
    phase2: Phase2Summary
    phase3: Phase3Summary
    metrics: MetricsSummary


# --- Trace archive --------------------------------------------------------

TRACES_DIR = Path("traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class ReasoningTrace(BaseModel):
    trace_id: str
    timestamp: datetime
    user_query: str
    run_summary: MyceliumRunSummary
    sandbox_result: Dict[str, Any] = Field(default_factory=dict)


def append_trace(trace: ReasoningTrace) -> None:
    """Append a trace as a JSON line in a date-partitioned file.

    This is intentionally simple for the PoC and can be replaced with a
    more robust store later (SQLite, vector DB, etc.).
    """

    day_file = TRACES_DIR / f"{trace.timestamp.date()}.jsonl"
    with day_file.open("a", encoding="utf-8") as f:
        f.write(trace.model_dump_json() + "\n")


def metrics_to_summary(metrics: WorkflowMetrics) -> MetricsSummary:
    data = metrics.to_dict()
    return MetricsSummary(
        layer0_routes=data.get("layer0_routes", {}),
        routing_classifications=data.get("routing_classifications", {}),
        expert_decisions=data.get("expert_decisions", {}),
        domains=data.get("domains", {}),
    )
