"""expert_system_init_api.py

FastAPI router that exposes async UnifiedExpertSystem initialisation
endpoints.

This module is a fire-and-poll HTTP job-queue wrapper whose only job is
to start the long-running UnifiedExpertSystem.__init__ in a background
thread so the frontend never hits the 300 s gateway timeout.  It has
**no connection to sklearn probability calibration** — see
``mycelium.pipeline.layer0.probability_calibration`` for that.

Previous name: ``calibration_api.py`` (misleading; renamed for clarity).
A backward-compatible re-export shim remains at the old path.

Endpoints
---------
POST /api/expert-system/init/start
    Start UnifiedExpertSystem initialisation as a background task.
    Returns ``{job_id, status: "pending"}`` immediately (< 1 ms).

GET  /api/expert-system/init/status/{job_id}
    Poll job status.  Returns:
    ``{job_id, status, progress, error}``
    status: "pending" | "running" | "complete" | "error"
    The frontend polls this every few seconds; no connection is held
    open long enough to hit the gateway timeout.

GET  /api/expert-system/ready
    Quick boolean readiness check.

Usage
-----
    from mycelium.pipeline.expert_system_init_api import expert_system_init_router
    app.include_router(expert_system_init_router)

The initialised UnifiedExpertSystem is stored on ``app.state.expert_system``
once the job completes so the rest of the application can access it via
    request.app.state.expert_system
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

expert_system_init_router = APIRouter()

_jobs: Dict[str, dict] = {}


def _new_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "progress": 0, "error": None}
    return job_id


def _blocking_init(job_id: str, app_state) -> None:
    """Synchronous UnifiedExpertSystem init — runs inside a thread-pool executor."""
    try:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["progress"] = 5

        raise ImportError("unified_expert_system removed in TRM v0.2 — use TRMV2InferenceEngine")  # noqa: PLC0415

        _jobs[job_id]["progress"] = 10
        system = UnifiedExpertSystem(
            enable_calibration=True,
            enable_ood_detection=True,
        )
        _jobs[job_id]["progress"] = 100

        app_state.expert_system = system
        _jobs[job_id]["status"] = "complete"

    except Exception as exc:  # noqa: BLE001
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


async def _run_init(job_id: str, app_state) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _blocking_init, job_id, app_state)


@expert_system_init_router.post(
    "/api/expert-system/init/start",
    summary="Start UnifiedExpertSystem initialisation",
)
async def start_init(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    job_id = _new_job()
    background_tasks.add_task(_run_init, job_id, request.app.state)
    return JSONResponse({"job_id": job_id, "status": "pending"})


@expert_system_init_router.get(
    "/api/expert-system/init/status/{job_id}",
    summary="Poll expert-system init job status",
)
async def init_status(job_id: str) -> JSONResponse:
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(
        {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "error": job.get("error"),
        }
    )


@expert_system_init_router.get(
    "/api/expert-system/ready",
    summary="Quick readiness check for expert system",
)
async def expert_system_ready(request: Request) -> JSONResponse:
    ready = getattr(request.app.state, "expert_system", None) is not None
    return JSONResponse({"ready": ready})
