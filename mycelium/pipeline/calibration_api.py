"""calibration_api.py — backward-compatibility shim.

This module has been renamed to ``expert_system_init_api.py`` because its
actual purpose — a fire-and-poll HTTP job-queue for UnifiedExpertSystem
initialisation — has nothing to do with sklearn probability calibration.

This shim re-exports everything under the old names so existing imports
(app.include_router, tests, RESTRUCTURING_PLAN.md references) continue
to work without any changes on the caller side.

For the actual sklearn classifier probability calibration, see:
    mycelium.pipeline.layer0.probability_calibration

This file may be removed once all callers have been updated to import
from ``expert_system_init_api`` directly.
"""

from mycelium.pipeline.expert_system_init_api import (  # noqa: F401
    expert_system_init_router as calibration_router,
    start_init as start_calibration,
    init_status as calibration_status,
    expert_system_ready as calibration_ready,
)

__all__ = [
    "calibration_router",
    "start_calibration",
    "calibration_status",
    "calibration_ready",
]
