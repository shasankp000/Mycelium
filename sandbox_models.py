from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from api_models import MyceliumRunSummary


class SandboxStep(BaseModel):
    tool: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    commentary: str
    status: str
    started_at: datetime
    finished_at: datetime


class SandboxTask(BaseModel):
    trace_id: str
    user_query: str
    hypotheses: List[str] = Field(default_factory=list)
    uncertainty_notes: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class SandboxResult(BaseModel):
    trace_id: str
    started_at: datetime
    finished_at: datetime
    steps: List[SandboxStep] = Field(default_factory=list)
    summary: str = ""


def build_sandbox_task_from_run(
    run: MyceliumRunSummary,
    *,
    user_query: str,
) -> SandboxTask:
    """Create a coarse SandboxTask from a Mycelium run.

    For now this uses simple heuristics and is mainly a data-shaping
    utility; smarter logic can be added later.
    """

    metrics_snapshot: Dict[str, Any] = {
        "layer0_routes": run.metrics.layer0_routes,
        "routing_classifications": run.metrics.routing_classifications,
        "expert_decisions": run.metrics.expert_decisions,
        "domains": run.metrics.domains,
    }

    hypotheses: List[str] = []
    if run.expert_decision.decision_type:
        hypotheses.append(
            f"Expert decision type: {run.expert_decision.decision_type}"
        )
    if run.phase3.validation_decision.get("result_class"):
        hypotheses.append(
            f"Validation result: {run.phase3.validation_decision['result_class']}"
        )

    now = datetime.utcnow()
    return SandboxTask(
        trace_id=run.trace_id,
        user_query=user_query,
        hypotheses=hypotheses,
        domains=list(run.routing.selected_domains),
        metrics_snapshot=metrics_snapshot,
        created_at=now,
    )
