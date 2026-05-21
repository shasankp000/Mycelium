"""calibration_api.py

FastAPI router that exposes async expert calibration endpoints.

Previously the frontend held an open HTTP connection for the entire
calibration pipeline (K-Medoids + OOD setup across all experts), which
regularly timed out at the 300 s gateway limit even when the work was
almost done.

This module replaces that pattern with a fire-and-poll job model:

  POST /api/calibrate/start
    \u2192 starts UnifiedExpertSystem initialisation as a background task
    \u2192 returns {job_id, status: "pending"} immediately (< 1 ms)

  GET  /api/calibrate/status/{job_id}
    \u2192 returns {job_id, status, progress, error}
    \u2192 status: "pending" | "running" | "complete" | "error"
    \u2192 frontend polls this every few seconds; no connection is ever held
      open long enough to hit the timeout

Usage in your main FastAPI app
------------------------------
    from mycelium.pipeline.calibration_api import calibration_router
    app.include_router(calibration_router)

The initialised UnifiedExpertSystem is stored on app.state.expert_system
once the job completes so the rest of the application can access it via
    request.app.state.expert_system
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

calibration_router = APIRouter()

_jobs: Dict[str, dict] = {}


def _new_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "progress": 0, "error": None}
    return job_id


def _blocking_calibration(job_id: str, app_state) -> None:
    """Synchronous calibration pipeline \u2014 runs inside a thread-pool executor."""
    try:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["progress"] = 5

        from mycelium.pipeline.unified_expert_system import UnifiedExpertSystem  # noqa: PLC0415

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


async def _run_calibration(job_id: str, app_state) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _blocking_calibration, job_id, app_state)


@calibration_router.post("/api/calibrate/start", summary="Start expert calibration")
async def start_calibration(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    job_id = _new_job()
    background_tasks.add_task(_run_calibration, job_id, request.app.state)
    return JSONResponse({"job_id": job_id, "status": "pending"})


@calibration_router.get(
    "/api/calibrate/status/{job_id}",
    summary="Poll calibration job status",
)
async def calibration_status(job_id: str) -> JSONResponse:
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


@calibration_router.get(
    "/api/calibrate/ready",
    summary="Quick readiness check",
)
async def calibration_ready(request: Request) -> JSONResponse:
    ready = getattr(request.app.state, "expert_system", None) is not None
    return JSONResponse({"ready": ready})
